import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PATCH_SIZE = 64              # ขนาดภาพสำหรับ CNN (64x64)
MIN_SKIN_RATIO = 0.70        # สัดส่วนเนื้อผิวหนังขั้นต่ำใน Patch
MAX_YAW_THRESHOLD = 0.25     # เกณฑ์มุมหันหน้า

# Landmark IDs สำหรับตำแหน่งบนใบหน้า (MediaPipe Face Mesh)
NOSE_ID = 1                  # ปลายจมูก
FOREHEAD_ID = 10             # กลางหน้าผาก
CHEEK_L_EDGE_ID = 234        # ขอบแก้มซ้าย
CHEEK_R_EDGE_ID = 454        # ขอบแก้มขวา
CHEEK_L_ID = 117             # โหนกแก้มซ้าย
CHEEK_R_ID = 346             # โหนกแก้มขวา

DATASET_DIR = "dataset"
OUTPUT_DIR = "cnn_dataset"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def is_valid_skin_color(patch_bgr):
    """
    ตรวจสอบว่า Patch เป็นสีผิวคนจริง (ตัดสีเขียว/ฟ้า/มืด/แสงสะท้อนจ้า)
    """
    if patch_bgr is None:
        return False
        
    h_patch, w_patch = patch_bgr.shape[:2]
    if h_patch != PATCH_SIZE or w_patch != PATCH_SIZE:
        return False

    # แยก Channel ด้วย cv2.split
    b_chan, g_chan, r_chan = cv2.split(patch_bgr)

    r_mean = float(np.mean(r_chan))
    g_mean = float(np.mean(g_chan))
    b_mean = float(np.mean(b_chan))

    # 1. กฎผิวคน: สีแดงต้องเด่นกว่าสีน้ำเงิน
    if b_mean >= r_mean:
        return False

    # 2. ป้องกันภาพมืดจัด หรือขาวจ้าเกินไป
    if r_mean < 40 or (r_mean > 245 and g_mean > 245 and b_mean > 245):
        return False

    # 3. ตรวจสอบเม็ดสีผิวผ่าน YCrCb
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)

    skin_mask = (cr_chan >= 133) & (cr_chan <= 173) & (cb_chan >= 77) & (cb_chan <= 127)
    skin_ratio = float(np.sum(skin_mask)) / float(PATCH_SIZE * PATCH_SIZE)

    return skin_ratio >= MIN_SKIN_RATIO


def extract_square_patch(image_bgr, center_x, center_y, patch_size=64):
    """
    Crop ภาพสี่เหลี่ยมจัตุรัส patch_size x patch_size โดยไม่บิดสัดส่วน
    """
    h, w = image_bgr.shape[:2]
    half = patch_size // 2

    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None

    return image_bgr[y1:y2, x1:x2].copy()


# ==========================================
# MAIN EXTRACTION PIPELINE
# ==========================================
def process_dataset():
    classes = ["Normal", "Flushing"]
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for class_name in classes:
        in_class_dir = os.path.join(DATASET_DIR, class_name)
        out_class_dir = os.path.join(OUTPUT_DIR, class_name)

        if not os.path.exists(in_class_dir):
            continue
        os.makedirs(out_class_dir, exist_ok=True)

        image_files = [f for f in os.listdir(in_class_dir) if f.lower().endswith(valid_extensions)]
        saved_count = 0

        print(f"[*] Processing Class: {class_name} ({len(image_files)} files)")

        for filename in image_files:
            file_path = os.path.join(in_class_dir, filename)
            
            frame_bgr = cv2.imread(file_path)
            if frame_bgr is None:
                continue

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                continue

            # ดึง Landmarks ของใบหน้าแรก
            landmarks = None
            for face in results.multi_face_landmarks:
                landmarks = face.landmark
                break

            if landmarks is None:
                continue

            # 1. ประเมินมุมหันของศีรษะ (Head Yaw)
            nose_x = landmarks[NOSE_ID].x
            left_cheek_edge_x = landmarks[CHEEK_L_EDGE_ID].x
            right_cheek_edge_x = landmarks[CHEEK_R_EDGE_ID].x

            dist_left = abs(nose_x - left_cheek_edge_x)
            dist_right = abs(right_cheek_edge_x - nose_x)
            total_span = dist_left + dist_right

            skip_left = False
            skip_right = False

            if total_span > 0:
                if (dist_left / total_span) < MAX_YAW_THRESHOLD:
                    skip_left = True
                if (dist_right / total_span) < MAX_YAW_THRESHOLD:
                    skip_right = True

            # 2. จุดศูนย์กลางของแต่ละโซน
            forehead_pt = landmarks[FOREHEAD_ID]
            left_cheek_pt = landmarks[CHEEK_L_ID]
            right_cheek_pt = landmarks[CHEEK_R_ID]

            zones_to_extract = []
            zones_to_extract.append(("Forehead", forehead_pt.x * w, forehead_pt.y * h))
            
            if not skip_left:
                zones_to_extract.append(("Zone_L", left_cheek_pt.x * w, left_cheek_pt.y * h))
            if not skip_right:
                zones_to_extract.append(("Zone_R", right_cheek_pt.x * w, right_cheek_pt.y * h))

            # 3. Crop และคัดกรองคุณภาพ
            base_name, _ = os.path.splitext(filename)

            for zone_name, cx, cy in zones_to_extract:
                patch = extract_square_patch(frame_bgr, cx, cy, patch_size=PATCH_SIZE)

                if is_valid_skin_color(patch):
                    out_name = f"{base_name}_{zone_name}.jpg"
                    out_path = os.path.join(out_class_dir, out_name)
                    cv2.imwrite(out_path, patch)
                    saved_count += 1

        print(f"[+] Finished {class_name}: Saved {saved_count} clean patches.")

if __name__ == "__main__":
    process_dataset()