import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, static_image_mode=True) # ปรับเป็นโหมดภาพนิ่ง

def get_cheek_color(image_path):
    """ฟังก์ชันอ่านรูปภาพ หาแก้ม และคืนค่าเฉลี่ย H และ S"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: หาไฟล์ {image_path} ไม่เจอ!")
        return None, None, None

    h_img, w_img, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"Error: จับใบหน้าในรูป {image_path} ไม่ได้!")
        return None, None, None

    landmarks = results.pose_landmarks.landmark
    nose = landmarks[0]
    r_eye = landmarks[5]; r_ear = landmarks[8]
    l_eye = landmarks[2]; l_ear = landmarks[7]

    # พิกัดแก้ม
    r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w_img)
    r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h_img)
    box_size = 30

    r_cheek_roi = image[max(0, r_cheek_y-box_size):min(h_img, r_cheek_y+box_size), 
                        max(0, r_cheek_x-box_size):min(w_img, r_cheek_x+box_size)]
    
    if r_cheek_roi.size != 0:
        hsv_roi = cv2.cvtColor(r_cheek_roi, cv2.COLOR_BGR2HSV)
        avg_h = np.mean(hsv_roi[:, :, 0])
        avg_s = np.mean(hsv_roi[:, :, 1])
        
        # วาดกรอบให้ดูด้วยว่าจับตรงไหน
        cv2.rectangle(image, (r_cheek_x-box_size, r_cheek_y-box_size), (r_cheek_x+box_size, r_cheek_y+box_size), (0, 255, 0), 2)
        return avg_h, avg_s, image
    return None, None, None

# ==========================================
# ส่วนทดสอบ (ใส่ชื่อไฟล์รูปภาพของคุณตรงนี้)
# ==========================================
normal_img_path = "E:\SmartTriageAI\Face_normal.png"   # รูปหน้าปกติ
symptom_img_path = "E:\SmartTriageAI\Face_red.png" # รูปตอนมีอาการ (ซีด/แดง)

print("--- AI SKIN COLOR TESTER ---")
base_h, base_s, img_normal = get_cheek_color(normal_img_path)

if base_h is not None:
    print(f"[1] NORMAL  -> Baseline S: {base_s:.2f}, Baseline H: {base_h:.2f}")
    
    symp_h, symp_s, img_symp = get_cheek_color(symptom_img_path)
    if symp_h is not None:
        print(f"[2] SYMPTOM -> Current  S: {symp_s:.2f}, Current  H: {symp_h:.2f}")
        
        # -----------------------------------------------------
        # ตรรกะใหม่! ใช้ "เปอร์เซ็นต์" แทนตัวเลขตายตัว
        # -----------------------------------------------------
        s_drop_percent = ((base_s - symp_s) / base_s) * 100 if base_s > 0 else 0
        h_shift = base_h - symp_h
        
        print(f"    -> S Dropped by : {s_drop_percent:.1f}%")
        print(f"    -> H Shifted by : {h_shift:.1f} units")
        
        # ปรับความไวตรงนี้ (เช่น ถ้าความสดสีผิวดรอปลงเกิน 20% ถือว่าหน้าซีด)
        if s_drop_percent > 20.0: 
            print("\n>>> RESULT: PALLOR (PALE) DETECTED! <<<")
        elif h_shift > 5.0 and symp_s > base_s * 1.3: # เฉดสีเปลี่ยน และความสดสีเพิ่มขึ้น 30%
            print("\n>>> RESULT: FLUSHING (RED) DETECTED! <<<")
        else:
            print("\n>>> RESULT: NORMAL SKIN <<<")

        # โชว์รูปให้ดูว่า AI สกัดสีถูกที่ไหม
        cv2.imshow("Normal (Baseline)", img_normal)
        cv2.imshow("Symptom (Tested)", img_symp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()