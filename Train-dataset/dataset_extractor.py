import cv2
import mediapipe as mp
import os
import csv
import math
import numpy as np

# กำหนด Path ของ Dataset
DATASET_DIR = "dataset"
CATEGORIES = ["Normal", "Pallor", "Flushing"]
OUTPUT_CSV = "skin_dataset.csv"

mp_pose = mp.solutions.pose

def calculate_fitzpatrick(l_val, b_val):
    """คำนวณ ITA และจัดกลุ่ม Fitzpatrick Scale"""
    # ป้องกัน error กรณี b_val เป็น 0
    if b_val == 0: b_val = 0.001 
    
    # OpenCV เก็บ L* ในช่วง 0-255 (เราต้องแปลงกลับเป็น 0-100 ตามมาตรฐาน CIELab)
    # OpenCV เก็บ a*, b* ในช่วง 0-255 (เราต้องแปลงกลับเป็น -127 ถึง 127)
    real_L = (l_val * 100) / 255.0
    real_B = b_val - 128.0

    # สมการ ITA
    ita = math.atan((real_L - 50.0) / real_B) * (180.0 / math.pi)

    if ita > 55: return ita, "Type_I"
    elif 41 < ita <= 55: return ita, "Type_II"
    elif 28 < ita <= 41: return ita, "Type_III"
    elif 10 < ita <= 28: return ita, "Type_IV"
    elif -30 < ita <= 10: return ita, "Type_V"
    else: return ita, "Type_VI"

def process_images():
    print(f"กำลังเริ่มสกัด Feature จากภาพในโฟลเดอร์ {DATASET_DIR}...")
    
    # เปิดไฟล์ CSV เพื่อเตรียมเขียนข้อมูล
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # เขียน Header ของตาราง
        writer.writerow(["Filename", "Label", "L_mean", "A_mean", "B_mean", "ITA_Angle", "Fitzpatrick"])

        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            for category in CATEGORIES:
                folder_path = os.path.join(DATASET_DIR, category)
                if not os.path.exists(folder_path):
                    print(f"⚠️ ไม่พบโฟลเดอร์: {folder_path} ข้ามการทำงาน...")
                    continue

                # วนลูปอ่านทุกไฟล์ในโฟลเดอร์
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, filename)
                        image = cv2.imread(img_path)
                        if image is None: continue

                        h, w, _ = image.shape
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)

                        # ถ้าเจอใบหน้า/โครงร่าง
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            nose = landmarks[0]
                            r_eye = landmarks[5]
                            r_ear = landmarks[8]

                            # คำนวณพิกัดแก้มขวา
                            r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
                            r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
                            box_size = 15

                            # ครอปพื้นที่แก้ม
                            roi = image[max(0, r_cheek_y-box_size):min(h, r_cheek_y+box_size),
                                        max(0, r_cheek_x-box_size):min(w, r_cheek_x+box_size)]

                            if roi.size != 0:
                                # แปลงเป็น CIELab
                                lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                                l_mean = np.mean(lab_roi[:, :, 0])
                                a_mean = np.mean(lab_roi[:, :, 1])
                                b_mean = np.mean(lab_roi[:, :, 2])

                                # คัดแยก Fitzpatrick
                                ita_angle, fitz_type = calculate_fitzpatrick(l_mean, b_mean)

                                # บันทึกลงตาราง
                                writer.writerow([filename, category, round(l_mean, 2), round(a_mean, 2), round(b_mean, 2), round(ita_angle, 2), fitz_type])
                                print(f"✅ ประมวลผล: {filename} | Type: {fitz_type} | Label: {category}")

    print(f"\n🎉 สกัดข้อมูลเสร็จสิ้น! บันทึกผลลัพธ์ไว้ที่: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_images()