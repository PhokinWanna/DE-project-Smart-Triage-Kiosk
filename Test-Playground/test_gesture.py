# test3
# import cv2
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)

# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             continue

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)

#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
            
#             # จุดอ้างอิง
#             right_shoulder = landmarks[12]
#             left_shoulder = landmarks[11]
            
#             # ขอบเขตหน้าอก (Chest Zone)
#             body_left_bound = left_shoulder.x
#             body_right_bound = right_shoulder.x
#             chest_top = left_shoulder.y
#             chest_bottom = left_shoulder.y + 0.4 # เพิ่มความยาวลงมาหน่อย

#             def is_chest_pain(wrist, shoulder, side_name):
#                 # 1. เช็ค Visibility (แก้ผีหลอก)
#                 # ถ้าความมั่นใจต่ำกว่า 60% ถือว่ามองไม่เห็นมือ -> ข้าม
#                 if wrist.visibility < 0.6:
#                     return False

#                 # 2. เช็คตำแหน่ง 2D (อยู่ในกล่องไหม)
#                 in_box_x = body_right_bound < wrist.x < body_left_bound
#                 in_box_y = chest_top < wrist.y < chest_bottom
                
#                 if not (in_box_x and in_box_y):
#                     return False

#                 # 3. เช็คความลึก Z-Axis (แก้ท่าตั้งการ์ด)
#                 # MediaPipe Z: ค่าลบ = ใกล้กล้อง, ค่าบวก = ไกลกล้อง
#                 # ถ้ามือยื่นมาข้างหน้าเยอะๆ ค่า Z มือ จะน้อยกว่า Z ไหล่ มากๆ
                
#                 # คำนวณความต่างของความลึก
#                 depth_diff = wrist.z - shoulder.z
                
#                 # Logic: ถ้ามือแนบอก ค่า depth_diff ควรจะใกล้เคียง 0 (เช่น -0.1 ถึง 0.1)
#                 # แต่ถ้าตั้งการ์ด มือจะยื่นมาข้างหน้า ค่าจะเป็นลบเยอะๆ (เช่น -0.3, -0.4)
                
#                 # Threshold: ถ้ามือยื่นออกมาเกิน -0.15 ถือว่าตั้งการ์ด (ปรับเลขนี้ได้)
#                 is_touching_body = depth_diff > -0.15 
                
#                 # (Optional) Print ค่าเพื่อ Debug ดูว่าตั้งการ์ดค่า Z เป็นเท่าไหร่
#                 # print(f"{side_name} Depth Diff: {depth_diff:.3f}")

#                 return is_touching_body

#             # ตรวจสอบ
#             r_pain = is_chest_pain(landmarks[16], right_shoulder, "Right")
#             l_pain = is_chest_pain(landmarks[15], left_shoulder, "Left")

#             cv2.putText(image, f"R Depth: {landmarks[16].z:.3f}", (10, 80), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.putText(image, f"L Depth: {landmarks[15].z:.3f}", (10, 110), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

#             status_text = "Status: Normal"
#             color = (0, 255, 0)

#             if r_pain or l_pain:
#                 status_text = "⚠️ CHEST PAIN DETECTED!"
#                 color = (0, 0, 255)
#                 # วาดกล่องแจ้งเตือน
#                 h, w, _ = image.shape
#                 cv2.rectangle(image, 
#                               (int(body_right_bound * w), int(chest_top * h)), 
#                               (int(body_left_bound * w), int(chest_bottom * h)), 
#                               (0, 0, 255), 2)

#             cv2.putText(image, status_text, (10, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
#             # วาดเส้นปกติ
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # cv2.imshow('Smart Triage V3 - Depth Check', cv2.flip(image, 1))
#         cv2.imshow('Smart Triage V3 - Depth Check', image)
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()












# test4
import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ฟังก์ชันคำนวณระยะห่างแบบ 2D (Pythagoras)
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

# Tips: model_complexity=1 (ปานกลาง) หรือ 2 (แม่นสุดแต่ช้า) เลือกเอาตามสเปคเครื่อง
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # Pre-process
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- 1. Define Landmarks ---
            r_shoulder = landmarks[12]
            l_shoulder = landmarks[11]
            r_wrist = landmarks[16]
            l_wrist = landmarks[15]
            
            # จุดกึ่งกลางระหว่างไหล่ (Sternum / กระดูกกลางอก)
            mid_shoulder_x = (r_shoulder.x + l_shoulder.x) / 2
            mid_shoulder_y = (r_shoulder.y + l_shoulder.y) / 2
            
            # --- 2. Calculate Body Scale (The Ruler) ---
            # ความกว้างไหล่คือ "หน่วยวัด" ของเราในเฟรมนี้
            shoulder_width = calculate_distance(r_shoulder, l_shoulder)
            
            # กันเหนียว: ถ้าหาไหล่ไม่เจอ หรือตัวตะแคงข้างจนไหล่ทับกัน (width ~ 0) ให้ข้าม
            if shoulder_width < 0.01: 
                continue
            avg_shoulder_z = (r_shoulder.z + l_shoulder.z) / 2
            
            # --- 3. The "Cylinder" Logic Check ---
            def check_hand_on_chest(wrist, side_name):
                # A. Visibility Check (ต้องเห็นมือชัด)
                if wrist.visibility < 0.6: return False

                # B. Vertical Check (แกน Y)
                # มือต้องอยู่ต่ำกว่าไหล่ และสูงกว่า (ไหล่ + 1.2 เท่าของความกว้างไหล่) -> ประมาณลิ้นปี่
                # ใช้ shoulder_width มากำหนดขอบเขต แทนการใช้ค่าคงที่
                y_min = mid_shoulder_y
                y_max = mid_shoulder_y + (shoulder_width * 1.5) 
                
                if not (y_min < wrist.y < y_max):
                    return False
                
                # C. Horizontal Check (แกน X - ระยะห่างจากแกนกลาง)
                # มือต้องห่างจากจุดกึ่งกลางอก ไม่เกิน 0.8 เท่าของความกว้างไหล่
                # (ถ้าเกินแปลว่ากางแขนออก)
                dist_from_center_x = abs(wrist.x - mid_shoulder_x)
                if dist_from_center_x > (shoulder_width * 0.8):
                    return False

                # D. Depth Check (แกน Z - หัวใจสำคัญ!) 
                # สูตร: Z_Score = (Z_มือ - Z_ไหล่) / ความกว้างไหล่
                # ค่า Z ของ MediaPipe:
                # - ค่าลบเยอะๆ = ยื่นมาข้างหน้า (Boxing)
                # - ค่าใกล้ 0 = ระนาบเดียวกับตัว (Touching)
                # - ค่าบวก = อยู่หลังตัว
                
                # เราใช้ค่าเฉลี่ย Z ของไหล่ทั้งสองข้างเป็นระนาบอ้างอิง
                # avg_shoulder_z = (r_shoulder.z + l_shoulder.z) / 2
                raw_z_diff = wrist.z - avg_shoulder_z
                
                # Normalize ด้วย shoulder_width (Scale Invariant!)
                z_score = raw_z_diff / shoulder_width
                
                # --- CALIBRATION ZONE ---
                # จากการทดลอง:
                # ท่าแนบอก: z_score จะอยู่ประมาณ -0.5 ถึง 0.1
                # ท่าตั้งการ์ด: z_score จะน้อยกว่า -0.8 (ยื่นไปข้างหน้าเยอะเมื่อเทียบกับตัว)
                
                # เราตั้ง Threshold ที่ -0.6
                # (แปลว่า ยื่นมือออกมาได้ไม่เกิน 60% ของความกว้างไหล่ตัวเอง)
                is_touching_depth = z_score < -3.125
                
                # Debug Print (เอาไว้ดูค่าตอนจูน)
                print(f"{side_name} Z-Score: {z_score:.2f}")
                
                return is_touching_depth

            # Check Both Hands
            r_pain = check_hand_on_chest(r_wrist, "Right")
            l_pain = check_hand_on_chest(l_wrist, "Left")

            # --- Visualization ---
            status_text = "Normal"
            color = (0, 255, 0)
            
            if r_pain or l_pain:
                status_text = "⚠️ CHEST PAIN DETECTED!"
                color = (0, 0, 255)
                # วาดวงกลมที่กลางอกเพื่อบอกว่า Detect เจอ
                h, w, _ = image.shape
                cx, cy = int(mid_shoulder_x * w), int(mid_shoulder_y * h)
                cv2.circle(image, (cx, cy), 15, (0, 0, 255), -1)

            # Show Info
            cv2.putText(image, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Scale(Shoulder): {shoulder_width:.3f} and {avg_shoulder_z:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Smart Triage V4 - Scale Invariant',image)#, cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()