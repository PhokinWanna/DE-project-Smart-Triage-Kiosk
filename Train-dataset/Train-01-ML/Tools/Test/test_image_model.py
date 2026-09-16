import cv2
import mediapipe as mp
import numpy as np
import math
import joblib
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'skin_rf_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'skin_scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'skin_encoder.pkl')

BOX_SIZE_RATIO = 0.12          
Y_OFFSET_RATIO = 0.05          
CHEEK_VISIBILITY_RATIO = 0.3   

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
        real_B = 0.001
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
        if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
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

def process_single_image(image_path, model, scaler, encoder, face_mesh):
    """ฟังก์ชันสำหรับเทสรูปภาพ 1 รูป"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ หาไฟล์ไม่เจอ: {image_path}")
        return

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    status_text = "No face detected!"
    color = COLOR_WHITE
    a_mean, ita_angle = 0, 0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        rois, boxes = get_rois(landmarks, image, w, h)

        # วาดกรอบ
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)

        if rois:
            # รูปนิ่งมีแค่เฟรมเดียว เราใช้ raw_status ตรงๆ เลย ไม่ต้องมี Buffer
            raw_status, a_mean, ita_angle = classify_frame(rois, model, scaler, encoder)
            status_text = raw_status
            
            if status_text == "Flushing":
                color = COLOR_FLUSHING
            elif "WARNING" in status_text:
                color = COLOR_WARNING
            else:
                color = COLOR_NORMAL
        else:
            status_text = "Face turned away or clipped!"
            color = COLOR_TURNED_AWAY

    # เขียนข้อความบนรูปภาพ
    cv2.putText(image, f"Status: {status_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if a_mean > 0:
        cv2.putText(image, f"A* Mean: {a_mean:.1f} | ITA: {ita_angle:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # ปริ้นท์ค่าลง Terminal เพื่อให้วิเคราะห์ง่ายๆ
    print("\n" + "="*40)
    print(f"File: {os.path.basename(image_path)}")
    print(f"-> A* (Redness): {a_mean:.2f}")
    print(f"-> ITA Angle   : {ita_angle:.2f}")
    print(f"-> RESULT      : {status_text}")
    print("="*40)

    # โชว์รูปค้างไว้ (กดปุ่มอะไรก็ได้เพื่อดูรูปต่อไป)
    cv2.imshow(f'Analysis: {os.path.basename(image_path)}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    model, scaler, encoder = load_ai()
    mp_face_mesh = mp.solutions.face_mesh

    # --- ใส่ Path รูปภาพที่คุณต้องการเทสตรงนี้ ---
    TEST_IMAGES = [
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\man-6862921_1280.jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images.jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (1).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (2).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (3).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (4).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (6).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (7).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (8).jpg",
        "E:\SmartTriageAI\Train-dataset\Train-01-ML\images (9).jpg",
    ]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        for img_path in TEST_IMAGES:
            process_single_image(img_path, model, scaler, encoder, face_mesh)

if __name__ == "__main__":
    main()