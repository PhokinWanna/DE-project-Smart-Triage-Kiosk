# import cv2
# import mediapipe as mp
# import os
# import csv
# import math
# import numpy as np

# DATASET_DIR = "dataset"
# CATEGORIES = ["Normal", "Flushing"] 
# OUTPUT_CSV = "skin_dataset_v3.csv" # เปลี่ยนชื่อไฟล์กันสับสน

# mp_face_mesh = mp.solutions.face_mesh

# def calculate_fitzpatrick(l_val, b_val):
#     if b_val == 0: b_val = 0.001 
#     real_L = (l_val * 100) / 255.0
#     real_B = b_val - 128.0
#     ita = math.atan((real_L - 50.0) / real_B) * (180.0 / math.pi)

#     if ita > 55: return ita, "Type_I"
#     elif 41 < ita <= 55: return ita, "Type_II"
#     elif 28 < ita <= 41: return ita, "Type_III"
#     elif 10 < ita <= 28: return ita, "Type_IV"
#     elif -30 < ita <= 10: return ita, "Type_V"
#     else: return ita, "Type_VI"

# def process_images():
#     print(f"กำลังเริ่มสกัด Feature (ROI ใหญ่ 22% + ดึงหลบตา) จากโฟลเดอร์ {DATASET_DIR}...")
    
#     with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Filename", "Label", "L_mean", "A_mean", "B_mean", "ITA_Angle", "Fitzpatrick"])

#         with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
#             for category in CATEGORIES:
#                 folder_path = os.path.join(DATASET_DIR, category)
#                 if not os.path.exists(folder_path): continue

#                 for filename in os.listdir(folder_path):
#                     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         img_path = os.path.join(folder_path, filename)
#                         image = cv2.imread(img_path)
#                         if image is None: continue

#                         h, w, _ = image.shape
#                         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                         results = face_mesh.process(image_rgb)

#                         if not results.multi_face_landmarks:
#                             print(f"❌ [REJECT] หาหน้าไม่เจอ: {filename}")
#                             continue

#                         landmarks = results.multi_face_landmarks[0].landmark
#                         nose = landmarks[1]
#                         l_ear = landmarks[454]
#                         r_ear = landmarks[234]
#                         l_eye = landmarks[263]
#                         r_eye = landmarks[33]
#                         forehead = landmarks[151]

#                         # --- อัปเดตขนาดและตำแหน่งให้ตรงกับโค้ดตอนใช้งานจริง ---
#                         face_width_px = abs((r_ear.x - l_ear.x) * w)
#                         box_size = int(face_width_px * 0.22) # กรอบใหญ่ 22%
#                         y_offset = int(h * 0.05) # ดึงลง 5% หลบถุงใต้ตา

#                         r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
#                         r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h) + y_offset
                        
#                         l_cheek_x = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
#                         l_cheek_y = int((l_eye.y + nose.y + l_ear.y) / 3 * h) + y_offset

#                         f_x, f_y = int(forehead.x * w), int(forehead.y * h)

#                         # ตัดภาพ (ใช้ Slicing ป้องกัน index ทะลุขอบ)
#                         y1_r, y2_r = max(0, r_cheek_y-box_size), min(h, r_cheek_y+box_size)
#                         x1_r, x2_r = max(0, r_cheek_x-box_size), min(w, r_cheek_x+box_size)
#                         r_roi = image[y1_r:y2_r, x1_r:x2_r]

#                         y1_l, y2_l = max(0, l_cheek_y-box_size), min(h, l_cheek_y+box_size)
#                         x1_l, x2_l = max(0, l_cheek_x-box_size), min(w, l_cheek_x+box_size)
#                         l_roi = image[y1_l:y2_l, x1_l:x2_l]

#                         y1_f, y2_f = max(0, f_y-box_size), min(h, f_y+box_size)
#                         x1_f, x2_f = max(0, f_x-box_size), min(w, f_x+box_size)
#                         f_roi = image[y1_f:y2_f, x1_f:x2_f]

#                         # ถ้ารูปเล็กไปจนสกัดไม่ได้ ให้ทิ้ง
#                         if r_roi.size == 0 or l_roi.size == 0 or f_roi.size == 0:
#                             print(f"⚠️ [REJECT] ROI ล้นขอบเขต: {filename}")
#                             continue

#                         # รวมภาพแนวนอน
#                         combined_roi = np.concatenate((r_roi, l_roi, f_roi), axis=1)
                        
#                         # แปลงและหาค่าเฉลี่ย
#                         lab_roi = cv2.cvtColor(combined_roi, cv2.COLOR_BGR2LAB)
#                         l_mean = np.mean(lab_roi[:, :, 0])
#                         a_mean = np.mean(lab_roi[:, :, 1])
#                         b_mean = np.mean(lab_roi[:, :, 2])

#                         ita_angle, fitz_type = calculate_fitzpatrick(l_mean, b_mean)

#                         writer.writerow([filename, category, round(l_mean, 2), round(a_mean, 2), round(b_mean, 2), round(ita_angle, 2), fitz_type])
#                         print(f"✅ [SUCCESS] {filename} | A*: {a_mean:.2f} | Type: {fitz_type}")

#     print(f"\n🎉 สกัด Dataset V3 สำเร็จ! บันทึกที่: {OUTPUT_CSV}")

# if __name__ == "__main__":
#     process_images()

import cv2
import mediapipe as mp
import os
import csv
import math
import numpy as np

DATASET_DIR = "dataset"
CATEGORIES = ["Normal", "Flushing"] 
OUTPUT_CSV = "skin_dataset_v2.csv" 

mp_face_mesh = mp.solutions.face_mesh

def calculate_fitzpatrick(l_val, b_val):
    real_L = (l_val * 100) / 255.0
    real_B = b_val - 128.0
    if real_B == 0:
        real_B = 0.001  # avoid division by zero (fixed: was checking b_val==0 instead of real_B==0)

    ita = math.atan((real_L - 50.0) / real_B) * (180.0 / math.pi)

    if ita > 55: return ita, "Type_I"
    elif 41 < ita <= 55: return ita, "Type_II"
    elif 28 < ita <= 41: return ita, "Type_III"
    elif 10 < ita <= 28: return ita, "Type_IV"
    elif -30 < ita <= 10: return ita, "Type_V"
    else: return ita, "Type_VI"

def process_images():
    print(f"กำลังเริ่มสกัด Feature (V4: Head Pose Auto-Filter) จาก {DATASET_DIR}...")
    
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Label", "L_mean", "A_mean", "B_mean", "ITA_Angle", "Fitzpatrick", "Used_ROIs"])

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
            for category in CATEGORIES:
                folder_path = os.path.join(DATASET_DIR, category)
                if not os.path.exists(folder_path): continue

                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, filename)
                        image = cv2.imread(img_path)
                        if image is None: continue

                        h, w, _ = image.shape
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(image_rgb)

                        if not results.multi_face_landmarks:
                            print(f"❌ [REJECT] หาหน้าไม่เจอ: {filename}")
                            continue

                        landmarks = results.multi_face_landmarks[0].landmark
                        nose = landmarks[1]
                        l_ear = landmarks[454]
                        r_ear = landmarks[234]
                        l_eye = landmarks[263]
                        r_eye = landmarks[33]
                        forehead = landmarks[151]

                        # 1. คำนวณ Head Pose (หันซ้าย-ขวา ก้ม-เงย)
                        dist_nose_to_l_ear = abs(l_ear.x - nose.x)
                        dist_nose_to_r_ear = abs(r_ear.x - nose.x)

                        # box_size scaled from the full landmark bounding box (not just
                        # nose->ear x-distance), so it stays accurate as the head turns
                        xs = [lm.x for lm in landmarks]
                        ys = [lm.y for lm in landmarks]
                        face_width_px = (max(xs) - min(xs)) * w
                        face_height_px = (max(ys) - min(ys)) * h
                        face_size_px = (face_width_px + face_height_px) / 2
                        box_size = int(face_size_px * 0.22)
                        box_dim = box_size * 2  # fixed target size for every ROI crop
                        y_offset = int(h * 0.05)

                        # --- 2. Logic ตัดสินใจความสมบูรณ์ของแต่ละ ROI ---
                        valid_rois = []
                        used_parts = []

                        # แก้มขวา
                        if dist_nose_to_r_ear > (dist_nose_to_l_ear * 0.3):
                            r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
                            r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h) + y_offset
                            y1, y2 = max(0, r_cheek_y-box_size), min(h, r_cheek_y+box_size)
                            x1, x2 = max(0, r_cheek_x-box_size), min(w, r_cheek_x+box_size)
                            if (y2-y1 > 0) and (x2-x1 > 0):
                                valid_rois.append(image[y1:y2, x1:x2])
                                used_parts.append("R_Cheek")

                        # แก้มซ้าย
                        if dist_nose_to_l_ear > (dist_nose_to_r_ear * 0.3):
                            l_cheek_x = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
                            l_cheek_y = int((l_eye.y + nose.y + l_ear.y) / 3 * h) + y_offset
                            y1, y2 = max(0, l_cheek_y-box_size), min(h, l_cheek_y+box_size)
                            x1, x2 = max(0, l_cheek_x-box_size), min(w, l_cheek_x+box_size)
                            if (y2-y1 > 0) and (x2-x1 > 0):
                                valid_rois.append(image[y1:y2, x1:x2])
                                used_parts.append("L_Cheek")

                        # หน้าผาก
                        if forehead.y < l_eye.y and forehead.y < r_eye.y:
                            f_x, f_y = int(forehead.x * w), int(forehead.y * h)
                            y1, y2 = max(0, f_y-box_size), min(h, f_y+box_size)
                            x1, x2 = max(0, f_x-box_size), min(w, f_x+box_size)
                            if (y2-y1 > 0) and (x2-x1 > 0):
                                valid_rois.append(image[y1:y2, x1:x2])
                                used_parts.append("Forehead")

                        # 3. ถ้าไม่มีส่วนไหนใช้ได้เลย ให้ทิ้งรูปนี้
                        if len(valid_rois) == 0:
                            print(f"⚠️ [REJECT] หันหน้าหนีกล้องเกินไป: {filename}")
                            continue

                        # 4. Normalize ทุก ROI ให้เป็นขนาดเดียวกันก่อนต่อกัน
                        # (fixed: crops near the image edge get clipped to different
                        # heights, which used to crash np.concatenate on axis=1)
                        normalized_rois = [
                            cv2.resize(roi, (box_dim, box_dim))
                            for roi in valid_rois
                            if roi.shape[0] > 0 and roi.shape[1] > 0
                        ]

                        if len(normalized_rois) == 0:
                            print(f"⚠️ [REJECT] ROI ว่างเปล่าหลัง resize: {filename}")
                            continue

                        combined_roi = np.concatenate(normalized_rois, axis=1)

                        lab_roi = cv2.cvtColor(combined_roi, cv2.COLOR_BGR2LAB)
                        l_mean = np.mean(lab_roi[:, :, 0])
                        a_mean = np.mean(lab_roi[:, :, 1])
                        b_mean = np.mean(lab_roi[:, :, 2])

                        ita_angle, fitz_type = calculate_fitzpatrick(l_mean, b_mean)
                        parts_str = "+".join(used_parts)

                        writer.writerow([filename, category, round(l_mean, 2), round(a_mean, 2), round(b_mean, 2), round(ita_angle, 2), fitz_type, parts_str])
                        print(f"✅ [SUCCESS] {filename} | Used: {parts_str} | A*: {a_mean:.2f}")

    print(f"\n🎉 สกัด Dataset V4 สำเร็จ! บันทึกที่: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_images()