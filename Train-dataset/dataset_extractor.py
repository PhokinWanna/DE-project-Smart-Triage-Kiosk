import cv2
import mediapipe as mp
import os
import csv
import math
import numpy as np

DATASET_DIR = "dataset"
CATEGORIES = ["Normal", "Flushing"] # เพิ่ม Pallor ได้ถ้าหา Data ได้
OUTPUT_CSV = "skin_dataset_dynamic1.csv"

mp_face_mesh = mp.solutions.face_mesh

def calculate_fitzpatrick(l_val, b_val):
    if b_val == 0: b_val = 0.001 
    real_L = (l_val * 100) / 255.0
    real_B = b_val - 128.0
    ita = math.atan((real_L - 50.0) / real_B) * (180.0 / math.pi)

    if ita > 55: return ita, "Type_I"
    elif 41 < ita <= 55: return ita, "Type_II"
    elif 28 < ita <= 41: return ita, "Type_III"
    elif 10 < ita <= 28: return ita, "Type_IV"
    elif -30 < ita <= 10: return ita, "Type_V"
    else: return ita, "Type_VI"

def process_images():
    print(f"กำลังเริ่มสกัด Feature (Dynamic ROI - แก้มซ้าย, ขวา, หน้าผาก) จากโฟลเดอร์ {DATASET_DIR}...")
    
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Label", "L_mean", "A_mean", "B_mean", "ITA_Angle", "Fitzpatrick"])

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
                        
                        # พิกัดโครงสร้างหลัก
                        nose = landmarks[1]
                        l_ear = landmarks[454]
                        r_ear = landmarks[234]
                        l_eye = landmarks[263]
                        r_eye = landmarks[33]
                        forehead = landmarks[151] # จุดกึ่งกลางหน้าผาก

                        # DYNAMIC BOX SIZE: 12% ของความกว้างหน้า
                        face_width_px = abs((r_ear.x - l_ear.x) * w)
                        box_size = int(face_width_px * 0.12)

                        # คำนวณพิกัดกึ่งกลางของ 3 จุด (แก้มขวา, แก้มซ้าย, หน้าผาก)
                        r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
                        r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
                        
                        l_cheek_x = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
                        l_cheek_y = int((l_eye.y + nose.y + l_ear.y) / 3 * h)

                        f_x = int(forehead.x * w)
                        f_y = int(forehead.y * h)

                        # BOUNDARY CHECK: เช็คว่าขอบกล่องของทั้ง 3 จุดทะลุขอบรูปหรือไม่
                        if (r_cheek_x - box_size < 0 or r_cheek_x + box_size > w or r_cheek_y - box_size < 0 or r_cheek_y + box_size > h or
                            l_cheek_x - box_size < 0 or l_cheek_x + box_size > w or l_cheek_y - box_size < 0 or l_cheek_y + box_size > h or
                            f_x - box_size < 0 or f_x + box_size > w or f_y - box_size < 0 or f_y + box_size > h):
                            print(f"⚠️ [REJECT] หน้าแหว่ง/ล้นขอบภาพ (โดนตัดบริเวณแก้มหรือหน้าผาก): {filename}")
                            continue

                        # CROP ทั้ง 3 จุด
                        r_roi = image[r_cheek_y-box_size:r_cheek_y+box_size, r_cheek_x-box_size:r_cheek_x+box_size]
                        l_roi = image[l_cheek_y-box_size:l_cheek_y+box_size, l_cheek_x-box_size:l_cheek_x+box_size]
                        f_roi = image[f_y-box_size:f_y+box_size, f_x-box_size:f_x+box_size]

                        if r_roi.size != 0 and l_roi.size != 0 and f_roi.size != 0:
                            # รวมพิกเซลทั้ง 3 ส่วนเข้าด้วยกันแบบแนวนอน
                            combined_roi = np.concatenate((r_roi, l_roi, f_roi), axis=1)
                            
                            # แปลงเป็น CIELab และหาค่าเฉลี่ยจากพิกเซลทั้งหมด (3 จุดรวมกัน)
                            lab_roi = cv2.cvtColor(combined_roi, cv2.COLOR_BGR2LAB)
                            l_mean = np.mean(lab_roi[:, :, 0])
                            a_mean = np.mean(lab_roi[:, :, 1])
                            b_mean = np.mean(lab_roi[:, :, 2])

                            ita_angle, fitz_type = calculate_fitzpatrick(l_mean, b_mean)

                            writer.writerow([filename, category, round(l_mean, 2), round(a_mean, 2), round(b_mean, 2), round(ita_angle, 2), fitz_type])
                            print(f"✅ [SUCCESS] {filename} | A*: {a_mean:.2f} | Type: {fitz_type}")

    print(f"\n🎉 สกัด Dataset สำเร็จ! บันทึกที่: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_images()