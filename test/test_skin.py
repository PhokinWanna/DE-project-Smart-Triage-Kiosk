# # import cv2
# # import mediapipe as mp
# # import math
# # import time 
# # import numpy as np

# # mp_drawing = mp.solutions.drawing_utils
# # mp_pose = mp.solutions.pose

# # def calculate_angle(a, b, c):
# #     a = [a.x, a.y] 
# #     b = [b.x, b.y] 
# #     c = [c.x, c.y] 
# #     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
# #     angle = abs(radians*180.0/math.pi)
# #     if angle > 180.0: angle = 360-angle
# #     return angle

# # def ccw(A, B, C):
# #     return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# # def check_intersection(A, B, C, D):
# #     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# # cap = cv2.VideoCapture(0)

# # # ตัวแปรสำหรับจับเวลาท่าทาง
# # holding_start_time = 0
# # is_holding = False
# # active_zone = None 
# # CONFIRMATION_TIME = 2.0 

# # # ---------------------------------------------------------
# # # ตัวแปรสำหรับระบบ Skin Color (Calibration Option A)
# # # ---------------------------------------------------------
# # CALIBRATION_TIME = 3.0 # ใช้เวลา 3 วินาทีในการจำสีผิวเริ่มต้น
# # skin_calibrating = True
# # skin_calibration_start = 0
# # baseline_S = 0.0 # ค่าเฉลี่ย Saturation ปกติ
# # baseline_H = 0.0 # ค่าเฉลี่ย Hue ปกติ
# # S_history = []
# # H_history = []
# # current_skin_status = "NORMAL"

# # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# #     while cap.isOpened():
# #         success, image = cap.read()
# #         if not success: continue

# #         image = cv2.flip(image, 1)
# #         h, w, _ = image.shape 

# #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         image_rgb.flags.writeable = False
# #         results = pose.process(image_rgb)
# #         image.flags.writeable = True

# #         scratch_flag = [False]

# #         if results.pose_landmarks:
# #             landmarks = results.pose_landmarks.landmark
            
# #             nose = landmarks[0]
# #             mouth_l = landmarks[9] 
# #             mouth_r = landmarks[10]
# #             r_eye = landmarks[5] # ตาขวา
# #             l_eye = landmarks[2] # ตาซ้าย
# #             r_ear = landmarks[8]
# #             l_ear = landmarks[7]
            
# #             r_shoulder = landmarks[12]
# #             l_shoulder = landmarks[11]
# #             r_hip = landmarks[24] 
# #             l_hip = landmarks[23] 
            
# #             r_elbow = landmarks[14]
# #             r_wrist = landmarks[16]
# #             l_elbow = landmarks[13]
# #             l_wrist = landmarks[15]
# #             r_index = landmarks[20]
# #             l_index = landmarks[19]

# #             # ---------------------------------------------------------
# #             # FEATURE 5: Physiological Analysis (Skin Color Detect)
# #             # ---------------------------------------------------------
# #             if skin_calibrating and skin_calibration_start == 0:
# #                 skin_calibration_start = time.time()

# #             # สร้างพื้นที่แก้มซ้ายและขวา (Cheek ROI) แบบง่ายๆ จากจุดที่มีอยู่
# #             # แก้มขวา: กึ่งกลางระหว่างตาขวา, จมูก, และหูขวา
# #             r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
# #             r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
# #             # แก้มซ้าย: กึ่งกลางระหว่างตาซ้าย, จมูก, และหูซ้าย
# #             l_cheek_x = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
# #             l_cheek_y = int((l_eye.y + nose.y + l_ear.y) / 3 * h)
            
# #             box_size = 15 # ขนาดกรอบครอบแก้ม (Pixels)
            
# #             # วาดกรอบบนแก้มให้เห็นว่า AI กำลังจับสีตรงไหน
# #             cv2.rectangle(image, (r_cheek_x-box_size, r_cheek_y-box_size), (r_cheek_x+box_size, r_cheek_y+box_size), (255, 100, 100), 1)
# #             cv2.rectangle(image, (l_cheek_x-box_size, l_cheek_y-box_size), (l_cheek_x+box_size, l_cheek_y+box_size), (255, 100, 100), 1)

# #             # ตัดภาพ (Crop) บริเวณแก้มเพื่อนำไปวิเคราะห์
# #             r_cheek_roi = image[max(0, r_cheek_y-box_size):min(h, r_cheek_y+box_size), max(0, r_cheek_x-box_size):min(w, r_cheek_x+box_size)]
            
# #             if r_cheek_roi.size != 0:
# #                 # แปลงจาก BGR เป็น HSV (ดีที่สุดสำหรับการวิเคราะห์สีผิว)
# #                 hsv_roi = cv2.cvtColor(r_cheek_roi, cv2.COLOR_BGR2HSV)
# #                 avg_h = np.mean(hsv_roi[:, :, 0]) # Hue (เฉดสี)
# #                 avg_s = np.mean(hsv_roi[:, :, 1]) # Saturation (ความอิ่มตัว/ความซีด)

# #                 if skin_calibrating:
# #                     # ช่วง 3 วินาทีแรก เก็บค่าเฉลี่ย
# #                     S_history.append(avg_s)
# #                     H_history.append(avg_h)
                    
# #                     if time.time() - skin_calibration_start > CALIBRATION_TIME:
# #                         skin_calibrating = False
# #                         baseline_S = np.mean(S_history)
# #                         baseline_H = np.mean(H_history)
# #                         print(f"Calibration Done! Base S: {baseline_S:.1f}, Base H: {baseline_H:.1f}")
# #                 else:
# #                     # หลัง 3 วินาที วิเคราะห์เปรียบเทียบกับ Baseline
# #                     s_drop = baseline_S - avg_s  # ถ้าซีดลง ค่า S จะน้อยลง (drop เป็นบวก)
# #                     h_shift = baseline_H - avg_h # เทียบการเลื่อนของเฉดสี

# #                     print(f"Base SDrop: {s_drop:.1f}, Base Hshift: {h_shift:.1f}")

# #                     # ปรับตัวเลขเหล่านี้เพื่อความไว (Sensitivity) ของการจับหน้าซีด/หน้าแดง
# #                     if s_drop > 25: # สีผิวซีดจางลงอย่างมีนัยสำคัญ
# #                         current_skin_status = "PALLOR (PALE)"
# #                     elif h_shift > 10 and avg_s > baseline_S + 10: # เฉดสีแดงขึ้นและความอิ่มตัวสูงขึ้น
# #                         current_skin_status = "FLUSHING (RED)"
# #                     else:
# #                         current_skin_status = "NORMAL"

# #             # ---------------------------------------------------------
# #             # โค้ดส่วน Gesture Detection (V9.0 เดิม)
# #             # ---------------------------------------------------------
# #             r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
# #             l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
# #             r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
# #             l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
            
# #             chest_ratio = 0.55
# #             r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
# #             l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio
            
# #             chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
# #             cv2.polylines(image, [chest_pts], isClosed=True, color=(0, 255, 0), thickness=2)

# #             abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))
# #             cv2.polylines(image, [abd_pts], isClosed=True, color=(0, 165, 255), thickness=2)

# #             nose_px = np.array([nose.x * w, nose.y * h])
# #             shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
# #             base_head_w = shoulder_width * 0.48
# #             base_head_h = shoulder_width * 0.58
            
# #             mid_sh_px = (r_sh_px + l_sh_px) / 2
# #             offset_x = (nose_px[0] - mid_sh_px[0]) / (shoulder_width / 2) 
# #             offset_y = (nose_px[1] - (mid_sh_px[1] - base_head_h)) / base_head_h
            
# #             left_w = base_head_w * (1 + offset_x * 0.7)
# #             right_w = base_head_w * (1 - offset_x * 0.7)
# #             top_h = base_head_h * (1 - offset_y * 0.5)
# #             bot_h = base_head_h * 0.15 
            
# #             head_tl = nose_px + np.array([-left_w, -top_h])
# #             head_tr = nose_px + np.array([right_w, -top_h])
# #             head_br = nose_px + np.array([right_w, bot_h])
# #             head_bl = nose_px + np.array([-left_w, bot_h])
            
# #             head_pts = np.array([head_tl, head_tr, head_br, head_bl], np.int32).reshape((-1, 1, 2))
# #             cv2.polylines(image, [head_pts], isClosed=True, color=(0, 0, 255), thickness=3)

# #             r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h])
# #             r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
# #             r_index_px = np.array([r_index.x * w, r_index.y * h]) 
# #             l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h])
# #             l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
# #             l_index_px = np.array([l_index.x * w, l_index.y * h]) 
# #             mouth_px = np.array([(mouth_l.x + mouth_r.x)/2 * w, (mouth_l.y + mouth_r.y)/2 * h])

# #             mouth_threshold = shoulder_width * 0.30
# #             touching_mouth = (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or \
# #                              (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold) or \
# #                              (np.linalg.norm(r_index_px - mouth_px) < mouth_threshold) or \
# #                              (np.linalg.norm(l_index_px - mouth_px) < mouth_threshold)

# #             support_threshold = shoulder_width * 0.25
# #             supporting_elbow = (np.linalg.norm(r_wrist_px - l_elbow_px) < support_threshold) or \
# #                                (np.linalg.norm(r_index_px - l_elbow_px) < support_threshold) or \
# #                                (np.linalg.norm(l_wrist_px - r_elbow_px) < support_threshold) or \
# #                                (np.linalg.norm(l_index_px - r_elbow_px) < support_threshold)

# #             is_thinking_pose = touching_mouth or supporting_elbow

# #             r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
# #             l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
# #             r_angle_ok = 60 <= r_arm_angle <= 115
# #             l_angle_ok = 60 <= l_arm_angle <= 115
            
# #             arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
# #             is_arms_crossed = r_angle_ok and l_angle_ok and arms_intersect

# #             def get_touched_zone(wrist,elbow,shoulder,index_finger,ear):
# #                 if wrist.visibility < 0.5 and index_finger.visibility < 0.5: 
# #                     return None
# #                 arm_angle = calculate_angle(shoulder,elbow,wrist)
# #                 if arm_angle >= 115:
# #                     return None 
                
# #                 wrist_px = (int(wrist.x * w), int(wrist.y * h))
# #                 index_px = (int(index_finger.x *w), int(index_finger.y *h))
                
# #                 def is_in_poly(poly):
# #                     return (cv2.pointPolygonTest(poly,wrist_px, False) >=0) or \
# #                            (cv2.pointPolygonTest(poly, index_px, False) >= 0)

# #                 if is_in_poly(head_pts): 
# #                     if index_finger.z > (ear.z +0.02):
# #                         scratch_flag[0] =True 
# #                         return None
# #                     return "Head"
                
# #                 if is_in_poly(chest_pts): return "Chest"
# #                 if is_in_poly(abd_pts): return "Abdomen"
# #                 return None

# #             r_zone = get_touched_zone(r_wrist, r_elbow, r_shoulder, r_index, r_ear)
# #             l_zone = get_touched_zone(l_wrist, l_elbow, l_shoulder, l_index, l_ear)

# #             if is_arms_crossed or is_thinking_pose or scratch_flag[0]:
# #                 current_zone = None
# #             else:
# #                 current_zone = r_zone if r_zone else l_zone

# #             if current_zone:
# #                 if current_zone != active_zone:
# #                     active_zone = current_zone
# #                     holding_start_time = time.time()
# #                     is_holding = True
# #                 duration = time.time() - holding_start_time
# #             else:
# #                 is_holding = False
# #                 active_zone = None
# #                 holding_start_time = 0
# #                 duration = 0

# #             # ---------------------------------------------------------
# #             # NEW UI: AI TRIAGE MONITOR DASHBOARD (อัปเดตเพิ่มช่อง Skin)
# #             # ---------------------------------------------------------
# #             panel_w, panel_h = 350, 150 # ขยายความสูง Dashboard เพื่อใส่ข้อมูลผิว
# #             panel_x, panel_y = w - panel_w - 20, 20

# #             overlay = image.copy()
# #             cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
# #             cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

# #             cv2.putText(image, "Debug", (panel_x + 15, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
# #             cv2.line(image, (panel_x + 15, panel_y + 45), (panel_x + panel_w - 15, panel_y + 45), (100, 100, 100), 2)

# #             y_offset = panel_y + 75
            
# #             # --- ส่วนที่ 1: แสดงผล ท่าทาง (Posture) ---
# #             if is_arms_crossed:
# #                 cv2.putText(image, "Posture: IGNORED (Crossed Arms)", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
# #             elif is_thinking_pose:
# #                 cv2.putText(image, "Posture: IGNORED (Thinking)", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
# #             elif scratch_flag[0]:
# #                 cv2.putText(image, "Posture: IGNORED (Scratching)", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
# #             else:
# #                 if duration >= CONFIRMATION_TIME:
# #                     pulse = int(abs(math.sin(time.time() * 6)) * 255)
# #                     cv2.circle(image, (panel_x + panel_w - 30, panel_y + 25), 8, (0, 0, pulse), -1)
# #                     cv2.putText(image, f"Posture: ALERT ({active_zone})", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# #                 elif active_zone:
# #                     cv2.putText(image, f"Posture: ANALYZING {active_zone}...", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
# #                 else:
# #                     cv2.putText(image, "Posture: STANDBY (Normal)", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# #             # --- ส่วนที่ 2: แสดงผล สีผิว (Skin Physiological) ---
# #             y_skin = y_offset + 35
# #             if skin_calibrating:
# #                 cal_time_left = max(0, CALIBRATION_TIME - (time.time() - skin_calibration_start))
# #                 cv2.putText(image, f"Skin   : CALIBRATING ({cal_time_left:.1f}s)", (panel_x + 15, y_skin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
# #             else:
# #                 if current_skin_status == "PALLOR (PALE)":
# #                     cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 15, y_skin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # สีขาวซีด
# #                 elif current_skin_status == "FLUSHING (RED)":
# #                     cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 15, y_skin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # สีแดง
# #                 else:
# #                     cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 15, y_skin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# #             # --- ส่วนที่ 3: สรุปผลลัพธ์ที่จะส่งให้ LLM (Payload Preview) ---
# #             y_payload = y_skin + 35
# #             # cv2.putText(image, ">>> Ready for LLM Reasoning", (panel_x + 15, y_payload), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

# #             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# #         cv2.imshow('Smart Triage V10.0 - Full Vision Engine', image)
# #         if cv2.waitKey(5) & 0xFF == ord('q'): break

# # cap.release()
# # cv2.destroyAllWindows()








# import cv2
# import mediapipe as mp
# import math
# import time 
# import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def calculate_angle(a, b, c):
#     a = [a.x, a.y] 
#     b = [b.x, b.y] 
#     c = [c.x, c.y] 
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = abs(radians*180.0/math.pi)
#     if angle > 180.0: angle = 360-angle
#     return angle

# def ccw(A, B, C):
#     return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# def check_intersection(A, B, C, D):
#     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# cap = cv2.VideoCapture(0)

# holding_start_time = 0
# is_holding = False
# active_zone = None 
# CONFIRMATION_TIME = 2.0 

# CALIBRATION_TIME = 3.0
# skin_calibrating = True
# skin_calibration_start = 0
# baseline_S = 0.0
# baseline_H = 0.0
# S_history = []
# H_history = []
# current_skin_status = "NORMAL"

# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success: continue

#         image = cv2.flip(image, 1)
#         h, w, _ = image.shape 

#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_rgb.flags.writeable = False
#         results = pose.process(image_rgb)
#         image.flags.writeable = True

#         scratch_flag = [False]

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
            
#             nose = landmarks[0]
#             mouth_l = landmarks[9] 
#             mouth_r = landmarks[10]
#             r_eye = landmarks[5]
#             l_eye = landmarks[2]
#             r_ear = landmarks[8]
#             l_ear = landmarks[7]
            
#             r_shoulder = landmarks[12]
#             l_shoulder = landmarks[11]
#             r_hip = landmarks[24] 
#             l_hip = landmarks[23] 
            
#             r_elbow = landmarks[14]
#             r_wrist = landmarks[16]
#             l_elbow = landmarks[13]
#             l_wrist = landmarks[15]
#             r_index = landmarks[20]
#             l_index = landmarks[19]

#             if skin_calibrating and skin_calibration_start == 0:
#                 skin_calibration_start = time.time()

#             r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
#             r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
#             l_cheek_x = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
#             l_cheek_y = int((l_eye.y + nose.y + l_ear.y) / 3 * h)
            
#             box_size = 10  # ลดจาก 15 → 10
            
#             cv2.rectangle(image, (r_cheek_x-box_size, r_cheek_y-box_size), (r_cheek_x+box_size, r_cheek_y+box_size), (255, 100, 100), 1)
#             cv2.rectangle(image, (l_cheek_x-box_size, l_cheek_y-box_size), (l_cheek_x+box_size, l_cheek_y+box_size), (255, 100, 100), 1)

#             r_cheek_roi = image[max(0, r_cheek_y-box_size):min(h, r_cheek_y+box_size), max(0, r_cheek_x-box_size):min(w, r_cheek_x+box_size)]
            
#             if r_cheek_roi.size != 0:
#                 hsv_roi = cv2.cvtColor(r_cheek_roi, cv2.COLOR_BGR2HSV)
#                 avg_h = np.mean(hsv_roi[:, :, 0])
#                 avg_s = np.mean(hsv_roi[:, :, 1])

#                 if skin_calibrating:
#                     S_history.append(avg_s)
#                     H_history.append(avg_h)
                    
#                     if time.time() - skin_calibration_start > CALIBRATION_TIME:
#                         skin_calibrating = False
#                         baseline_S = np.mean(S_history)
#                         baseline_H = np.mean(H_history)
#                         print(f"Calibration Done! Base S: {baseline_S:.1f}, Base H: {baseline_H:.1f}")
#                 else:
#                     s_drop = baseline_S - avg_s
#                     h_shift = baseline_H - avg_h

#                     print(f"Base SDrop: {s_drop:.1f}, Base Hshift: {h_shift:.1f}")

#                     if s_drop > 25:
#                         current_skin_status = "PALLOR (PALE)"
#                     elif h_shift > 10 and avg_s > baseline_S + 10:
#                         current_skin_status = "FLUSHING (RED)"
#                     else:
#                         current_skin_status = "NORMAL"

#             r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
#             l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
#             r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
#             l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
            
#             chest_ratio = 0.55
#             r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
#             l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio
            
#             chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
#             cv2.polylines(image, [chest_pts], isClosed=True, color=(0, 255, 0), thickness=2)

#             abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))
#             cv2.polylines(image, [abd_pts], isClosed=True, color=(0, 165, 255), thickness=2)

#             nose_px = np.array([nose.x * w, nose.y * h])
#             shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
#             base_head_w = shoulder_width * 0.48
#             base_head_h = shoulder_width * 0.58
            
#             mid_sh_px = (r_sh_px + l_sh_px) / 2
#             offset_x = (nose_px[0] - mid_sh_px[0]) / (shoulder_width / 2) 
#             offset_y = (nose_px[1] - (mid_sh_px[1] - base_head_h)) / base_head_h
            
#             left_w = base_head_w * (1 + offset_x * 0.7)
#             right_w = base_head_w * (1 - offset_x * 0.7)
#             top_h = base_head_h * (1 - offset_y * 0.5)
#             bot_h = base_head_h * 0.15 
            
#             head_tl = nose_px + np.array([-left_w, -top_h])
#             head_tr = nose_px + np.array([right_w, -top_h])
#             head_br = nose_px + np.array([right_w, bot_h])
#             head_bl = nose_px + np.array([-left_w, bot_h])
            
#             head_pts = np.array([head_tl, head_tr, head_br, head_bl], np.int32).reshape((-1, 1, 2))
#             cv2.polylines(image, [head_pts], isClosed=True, color=(0, 0, 255), thickness=3)

#             r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h])
#             r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
#             r_index_px = np.array([r_index.x * w, r_index.y * h]) 
#             l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h])
#             l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
#             l_index_px = np.array([l_index.x * w, l_index.y * h]) 
#             mouth_px = np.array([(mouth_l.x + mouth_r.x)/2 * w, (mouth_l.y + mouth_r.y)/2 * h])

#             mouth_threshold = shoulder_width * 0.30
#             touching_mouth = (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or \
#                              (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold) or \
#                              (np.linalg.norm(r_index_px - mouth_px) < mouth_threshold) or \
#                              (np.linalg.norm(l_index_px - mouth_px) < mouth_threshold)

#             support_threshold = shoulder_width * 0.25
#             supporting_elbow = (np.linalg.norm(r_wrist_px - l_elbow_px) < support_threshold) or \
#                                (np.linalg.norm(r_index_px - l_elbow_px) < support_threshold) or \
#                                (np.linalg.norm(l_wrist_px - r_elbow_px) < support_threshold) or \
#                                (np.linalg.norm(l_index_px - r_elbow_px) < support_threshold)

#             is_thinking_pose = touching_mouth or supporting_elbow

#             r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
#             l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
#             r_angle_ok = 60 <= r_arm_angle <= 115
#             l_angle_ok = 60 <= l_arm_angle <= 115
            
#             arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
#             is_arms_crossed = r_angle_ok and l_angle_ok and arms_intersect

#             def get_touched_zone(wrist, elbow, shoulder, index_finger, ear):
#                 if wrist.visibility < 0.5 and index_finger.visibility < 0.5: 
#                     return None
#                 arm_angle = calculate_angle(shoulder, elbow, wrist)
#                 if arm_angle >= 115:
#                     return None 
                
#                 wrist_px = (int(wrist.x * w), int(wrist.y * h))
#                 index_px = (int(index_finger.x * w), int(index_finger.y * h))
                
#                 def is_in_poly(poly):
#                     return (cv2.pointPolygonTest(poly, wrist_px, False) >= 0) or \
#                            (cv2.pointPolygonTest(poly, index_px, False) >= 0)

#                 if is_in_poly(head_pts): 
#                     if index_finger.z > (ear.z + 0.02):
#                         scratch_flag[0] = True 
#                         return None
#                     return "Head"
                
#                 if is_in_poly(chest_pts): return "Chest"
#                 if is_in_poly(abd_pts): return "Abdomen"
#                 return None

#             r_zone = get_touched_zone(r_wrist, r_elbow, r_shoulder, r_index, r_ear)
#             l_zone = get_touched_zone(l_wrist, l_elbow, l_shoulder, l_index, l_ear)

#             if is_arms_crossed or is_thinking_pose or scratch_flag[0]:
#                 current_zone = None
#             else:
#                 current_zone = r_zone if r_zone else l_zone

#             if current_zone:
#                 if current_zone != active_zone:
#                     active_zone = current_zone
#                     holding_start_time = time.time()
#                     is_holding = True
#                 duration = time.time() - holding_start_time
#             else:
#                 is_holding = False
#                 active_zone = None
#                 holding_start_time = 0
#                 duration = 0

#             # ---------------------------------------------------------
#             # UI DASHBOARD — smaller panel & tighter spacing
#             # ---------------------------------------------------------
#             panel_w, panel_h = 290, 120  # ลดจาก 350×150 → 290×120
#             panel_x, panel_y = w - panel_w - 15, 15

#             overlay = image.copy()
#             cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
#             cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

#             cv2.putText(image, "Debug", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
#             cv2.line(image, (panel_x + 10, panel_y + 32), (panel_x + panel_w - 10, panel_y + 32), (100, 100, 100), 1)

#             y_offset = panel_y + 55  # ลดจาก +75 → +55

#             # --- Posture ---
#             font_scale = 0.48  # ลดจาก 0.6 → 0.48
#             if is_arms_crossed:
#                 cv2.putText(image, "Posture: IGNORED (Crossed Arms)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
#             elif is_thinking_pose:
#                 cv2.putText(image, "Posture: IGNORED (Thinking)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
#             elif scratch_flag[0]:
#                 cv2.putText(image, "Posture: IGNORED (Scratching)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
#             else:
#                 if duration >= CONFIRMATION_TIME:
#                     pulse = int(abs(math.sin(time.time() * 6)) * 255)
#                     cv2.circle(image, (panel_x + panel_w - 22, panel_y + 18), 6, (0, 0, pulse), -1)
#                     cv2.putText(image, f"Posture: ALERT ({active_zone})", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
#                 elif active_zone:
#                     cv2.putText(image, f"Posture: ANALYZING {active_zone}...", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
#                 else:
#                     cv2.putText(image, "Posture: STANDBY (Normal)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

#             # --- Skin ---
#             y_skin = y_offset + 28  # ลดจาก +35 → +28
#             if skin_calibrating:
#                 cal_time_left = max(0, CALIBRATION_TIME - (time.time() - skin_calibration_start))
#                 cv2.putText(image, f"Skin   : CALIBRATING ({cal_time_left:.1f}s)", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
#             else:
#                 if current_skin_status == "PALLOR (PALE)":
#                     cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
#                 elif current_skin_status == "FLUSHING (RED)":
#                     cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
#                 else:
#                     cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         cv2.imshow('Smart Triage V10.0 - Full Vision Engine', image)
#         if cv2.waitKey(5) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import math
import time 
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = [a.x, a.y] 
    b = [b.x, b.y] 
    c = [c.x, c.y] 
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_intersection(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

cap = cv2.VideoCapture(0)

holding_start_time = 0
is_holding = False
active_zone = None 
CONFIRMATION_TIME = 2.0 

CALIBRATION_TIME = 3.0
skin_calibrating = True
skin_calibration_start = 0
baseline_S = 0.0
baseline_H = 0.0
S_history = []
H_history = []
current_skin_status = "NORMAL"

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape 

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image.flags.writeable = True

        scratch_flag = [False]

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            nose = landmarks[0]
            mouth_l = landmarks[9] 
            mouth_r = landmarks[10]
            r_eye = landmarks[5]
            l_eye = landmarks[2]
            r_ear = landmarks[8]
            l_ear = landmarks[7]
            
            r_shoulder = landmarks[12]
            l_shoulder = landmarks[11]
            r_hip = landmarks[24] 
            l_hip = landmarks[23] 
            
            r_elbow = landmarks[14]
            r_wrist = landmarks[16]
            l_elbow = landmarks[13]
            l_wrist = landmarks[15]
            r_index = landmarks[20]
            l_index = landmarks[19]

            if skin_calibrating and skin_calibration_start == 0:
                skin_calibration_start = time.time()

            r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
            r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
            l_cheek_x = int((l_eye.x + nose.x + l_ear.x) / 3 * w)
            l_cheek_y = int((l_eye.y + nose.y + l_ear.y) / 3 * h)
            
            box_size = 10


            r_cheek_roi = image[max(0, r_cheek_y-box_size):min(h, r_cheek_y+box_size), max(0, r_cheek_x-box_size):min(w, r_cheek_x+box_size)]
            
            if r_cheek_roi.size != 0:
                hsv_roi = cv2.cvtColor(r_cheek_roi, cv2.COLOR_BGR2HSV)
                avg_h = np.mean(hsv_roi[:, :, 0])
                avg_s = np.mean(hsv_roi[:, :, 1])

                if skin_calibrating:
                    S_history.append(avg_s)
                    H_history.append(avg_h)
                    
                    if time.time() - skin_calibration_start > CALIBRATION_TIME:
                        skin_calibrating = False
                        baseline_S = np.mean(S_history)
                        baseline_H = np.mean(H_history)
                        print(f"Base S: {baseline_S:.1f}, Base H: {baseline_H:.1f}")
                else:
                    s_drop = baseline_S - avg_s
                    h_shift = baseline_H - avg_h

                    print(f"SDrop: {s_drop:.1f}, Hshift: {h_shift:.1f}")

                    if s_drop > 25:
                        current_skin_status = "Pallor (Pale)"
                    elif h_shift > 10 and avg_s > baseline_S + 10:
                        current_skin_status = "Flushing (Red)"
                    else:
                        current_skin_status = "Normal"

            r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
            l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
            r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
            l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
            
            chest_ratio = 0.55
            r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
            l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio
            
            # --- chest / abdomen / head polylines HIDDEN (removed draws) ---
            chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
            abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))

            nose_px = np.array([nose.x * w, nose.y * h])
            shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
            base_head_w = shoulder_width * 0.48
            base_head_h = shoulder_width * 0.58
            
            mid_sh_px = (r_sh_px + l_sh_px) / 2
            offset_x = (nose_px[0] - mid_sh_px[0]) / (shoulder_width / 2) 
            offset_y = (nose_px[1] - (mid_sh_px[1] - base_head_h)) / base_head_h
            
            left_w = base_head_w * (1 + offset_x * 0.7)
            right_w = base_head_w * (1 - offset_x * 0.7)
            top_h = base_head_h * (1 - offset_y * 0.5)
            bot_h = base_head_h * 0.15 
            
            head_tl = nose_px + np.array([-left_w, -top_h])
            head_tr = nose_px + np.array([right_w, -top_h])
            head_br = nose_px + np.array([right_w, bot_h])
            head_bl = nose_px + np.array([-left_w, bot_h])
            
            head_pts = np.array([head_tl, head_tr, head_br, head_bl], np.int32).reshape((-1, 1, 2))
            # --- head box polyline HIDDEN (removed draw) ---

            r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h])
            r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
            r_index_px = np.array([r_index.x * w, r_index.y * h]) 
            l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h])
            l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
            l_index_px = np.array([l_index.x * w, l_index.y * h]) 
            mouth_px = np.array([(mouth_l.x + mouth_r.x)/2 * w, (mouth_l.y + mouth_r.y)/2 * h])

            mouth_threshold = shoulder_width * 0.30
            touching_mouth = (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or \
                             (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold) or \
                             (np.linalg.norm(r_index_px - mouth_px) < mouth_threshold) or \
                             (np.linalg.norm(l_index_px - mouth_px) < mouth_threshold)

            support_threshold = shoulder_width * 0.25
            supporting_elbow = (np.linalg.norm(r_wrist_px - l_elbow_px) < support_threshold) or \
                               (np.linalg.norm(r_index_px - l_elbow_px) < support_threshold) or \
                               (np.linalg.norm(l_wrist_px - r_elbow_px) < support_threshold) or \
                               (np.linalg.norm(l_index_px - r_elbow_px) < support_threshold)

            is_thinking_pose = touching_mouth or supporting_elbow

            r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle_ok = 60 <= r_arm_angle <= 115
            l_angle_ok = 60 <= l_arm_angle <= 115
            
            arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
            is_arms_crossed = r_angle_ok and l_angle_ok and arms_intersect

            def get_touched_zone(wrist, elbow, shoulder, index_finger, ear):
                if wrist.visibility < 0.5 and index_finger.visibility < 0.5: 
                    return None
                arm_angle = calculate_angle(shoulder, elbow, wrist)
                if arm_angle >= 115:
                    return None 
                
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                index_px = (int(index_finger.x * w), int(index_finger.y * h))
                
                def is_in_poly(poly):
                    return (cv2.pointPolygonTest(poly, wrist_px, False) >= 0) or \
                           (cv2.pointPolygonTest(poly, index_px, False) >= 0)

                if is_in_poly(head_pts): 
                    if index_finger.z > (ear.z + 0.02):
                        scratch_flag[0] = True 
                        return None
                    return "Head"
                
                if is_in_poly(chest_pts): return "Chest"
                if is_in_poly(abd_pts): return "Abdomen"
                return None

            r_zone = get_touched_zone(r_wrist, r_elbow, r_shoulder, r_index, r_ear)
            l_zone = get_touched_zone(l_wrist, l_elbow, l_shoulder, l_index, l_ear)

            if is_arms_crossed or is_thinking_pose or scratch_flag[0]:
                current_zone = None
            else:
                current_zone = r_zone if r_zone else l_zone

            if current_zone:
                if current_zone != active_zone:
                    active_zone = current_zone
                    holding_start_time = time.time()
                    is_holding = True
                duration = time.time() - holding_start_time
            else:
                is_holding = False
                active_zone = None
                holding_start_time = 0
                duration = 0

            # ---------------------------------------------------------
            # UI DASHBOARD ONLY — no skeleton, no zone boxes
            # ---------------------------------------------------------
            panel_w, panel_h = 290, 120
            panel_x, panel_y = w - panel_w - 15, 15

            overlay = image.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            cv2.putText(image, "Debug", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.line(image, (panel_x + 10, panel_y + 32), (panel_x + panel_w - 10, panel_y + 32), (100, 100, 100), 1)

            y_offset = panel_y + 55
            font_scale = 0.48

            if is_arms_crossed:
                cv2.putText(image, "Posture: IGNORED (Crossed Arms)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
            elif is_thinking_pose:
                cv2.putText(image, "Posture: IGNORED (Thinking)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
            elif scratch_flag[0]:
                cv2.putText(image, "Posture: IGNORED (Scratching)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
            else:
                if duration >= CONFIRMATION_TIME:
                    pulse = int(abs(math.sin(time.time() * 6)) * 255)
                    cv2.circle(image, (panel_x + panel_w - 22, panel_y + 18), 6, (0, 0, pulse), -1)
                    cv2.putText(image, f"Posture: ALERT ({active_zone})", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
                elif active_zone:
                    cv2.putText(image, f"Posture: ANALYZING {active_zone}...", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
                else:
                    cv2.putText(image, "Posture: STANDBY (Normal)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

            y_skin = y_offset + 28
            if skin_calibrating:
                cal_time_left = max(0, CALIBRATION_TIME - (time.time() - skin_calibration_start))
                cv2.putText(image, f"Skin   : CALIBRATING ({cal_time_left:.1f}s)", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
            else:
                if current_skin_status == "PALLOR (PALE)":
                    cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                elif current_skin_status == "FLUSHING (RED)":
                    cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
                else:
                    cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

            # --- skeleton/landmarks HIDDEN (removed mp_drawing.draw_landmarks) ---

        cv2.imshow('Smart Triage V10.0 - Full Vision Engine', image)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()