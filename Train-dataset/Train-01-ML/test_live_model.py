import cv2
import mediapipe as mp
import numpy as np
import math
import joblib
import os
from collections import deque

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'skin_rf_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'skin_scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'skin_encoder.pkl')

BOX_SIZE_RATIO = 0.22          # ROI box size relative to face width
Y_OFFSET_RATIO = 0.05          # cheek ROI vertical offset relative to frame height
CHEEK_VISIBILITY_RATIO = 0.3   # min ratio of near-side/far-side ear distance to accept a cheek

FRAME_BUFFER_SIZE = 30         # ~1 sec of frames for temporal smoothing
FLUSHING_FRAME_THRESHOLD = 18  # ~60% of buffer

L_MEAN_BRIGHT_LIMIT = 210
L_MEAN_DARK_LIMIT = 40

BOX_COLOR = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_WARNING = (0, 255, 255)
COLOR_FLUSHING = (0, 0, 255)
COLOR_NORMAL = (0, 255, 0)
COLOR_TURNED_AWAY = (0, 165, 255)


def load_ai():
    print("กำลังโหลด AI Model...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("✅ โหลด Model สำเร็จ!")
        return model, scaler, encoder
    except Exception as e:
        print(f"❌ Error โหลดไฟล์ Model ไม่สำเร็จ: {e}")
        exit()


def calculate_ita(l_val, b_val):
    real_L = (l_val * 100) / 255.0
    real_B = b_val - 128.0
    if real_B == 0:
        real_B = 0.001  # avoid division by zero (guards the actual danger value)
    return math.atan((real_L - 50.0) / real_B) * (180.0 / math.pi)


def get_rois(landmarks, image, w, h):
    """Return in-bounds ROI crops and their pixel boxes for the current frame."""
    nose = landmarks[1]
    l_ear = landmarks[454]
    r_ear = landmarks[234]
    l_eye = landmarks[263]
    r_eye = landmarks[33]
    forehead = landmarks[151]

    dist_nose_to_l_ear = abs(l_ear.x - nose.x)
    dist_nose_to_r_ear = abs(r_ear.x - nose.x)
    max_half_width = max(dist_nose_to_l_ear, dist_nose_to_r_ear)
    face_width_px = max_half_width * 2 * w

    box_size = int(face_width_px * BOX_SIZE_RATIO)
    y_offset = int(h * Y_OFFSET_RATIO)

    candidates = []

    if dist_nose_to_r_ear > (dist_nose_to_l_ear * CHEEK_VISIBILITY_RATIO):
        cx = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
        cy = int((r_eye.y + nose.y + r_ear.y) / 3 * h) + y_offset
        candidates.append((cx, cy))

    if dist_nose_to_l_ear > (dist_nose_to_r_ear * CHEEK_VISIBILITY_RATIO):
        cx = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
        cy = int((l_eye.y + nose.y + l_ear.y) / 3 * h) + y_offset
        candidates.append((cx, cy))

    if forehead.y < l_eye.y and forehead.y < r_eye.y:
        cx, cy = int(forehead.x * w), int(forehead.y * h)
        candidates.append((cx, cy))

    rois, boxes = [], []
    for cx, cy in candidates:
        y1, y2 = cy - box_size, cy + box_size
        x1, x2 = cx - box_size, cx + box_size
        if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:  # strict: reject, don't clip
            rois.append(image[y1:y2, x1:x2])
            boxes.append((x1, y1, x2, y2))

    return rois, boxes


def classify_frame(rois, model, scaler, encoder):
    """Return (status, a_mean, ita_angle) for one frame's ROI set."""
    combined_roi = np.concatenate(rois, axis=1)
    lab_roi = cv2.cvtColor(combined_roi, cv2.COLOR_BGR2LAB)

    l_mean = np.mean(lab_roi[:, :, 0])
    a_mean = np.mean(lab_roi[:, :, 1])
    b_mean = np.mean(lab_roi[:, :, 2])
    ita_angle = calculate_ita(l_mean, b_mean)

    if l_mean > L_MEAN_BRIGHT_LIMIT:
        return "WARNING: Too Bright!", a_mean, ita_angle
    if l_mean < L_MEAN_DARK_LIMIT:
        return "WARNING: Too Dark!", a_mean, ita_angle

    features = np.array([[a_mean, b_mean, ita_angle]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return encoder.inverse_transform(prediction)[0], a_mean, ita_angle


def main():
    model, scaler, encoder = load_ai()
    mp_face_mesh = mp.solutions.face_mesh

    print("🎥 กำลังเปิดกล้อง...")
    cap = cv2.VideoCapture(0)  # เปลี่ยนเป็น 1 ถ้าใช้กล้องต่อแยก

    # Fixed-size ring buffer: O(1) push + auto-evict instead of list + pop(0) (O(n))
    status_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    flushing_count = 0  # maintained incrementally: O(1)/frame instead of .count() O(n)/frame

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            status_text = "Searching for face..."
            color = COLOR_WHITE
            a_mean, ita_angle = 0, 0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                rois, boxes = get_rois(landmarks, image, w, h)

                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 1)

                if rois:
                    raw_status, a_mean, ita_angle = classify_frame(rois, model, scaler, encoder)

                    # account for the entry about to be evicted before it's pushed out
                    if len(status_buffer) == FRAME_BUFFER_SIZE and status_buffer[0] == "Flushing":
                        flushing_count -= 1
                    status_buffer.append(raw_status)
                    if raw_status == "Flushing":
                        flushing_count += 1

                    if flushing_count > FLUSHING_FRAME_THRESHOLD:
                        status_text, color = "Flushing (RED)", COLOR_FLUSHING
                    elif "WARNING" in raw_status:
                        status_text, color = raw_status, COLOR_WARNING
                    else:
                        status_text, color = "Normal", COLOR_NORMAL
                else:
                    status_text, color = "Face turned away!", COLOR_TURNED_AWAY

            cv2.putText(image, f"Status: {status_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if a_mean > 0:
                cv2.putText(image, f"A* Mean: {a_mean:.1f} | ITA: {ita_angle:.1f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            cv2.imshow('AI Skin Classifier - Realtime Test', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()