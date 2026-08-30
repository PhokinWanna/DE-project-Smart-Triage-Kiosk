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

            r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
            l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
            r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
            l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
            
            chest_ratio = 0.55
            r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
            l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio
            
            chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [chest_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, "Chest", tuple(l_sh_px.astype(int) - [0, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [abd_pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(image, "Abdomen", tuple(l_chest_bottom.astype(int) - [0, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

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
            cv2.polylines(image, [head_pts], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.putText(image, "Dynamic Head", tuple(head_tl.astype(int) - [0, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h])
            r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
            r_index_px = np.array([r_index.x * w, r_index.y * h]) 
            
            l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h])
            l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
            l_index_px = np.array([l_index.x * w, l_index.y * h]) 
            
            mouth_px = np.array([(mouth_l.x + mouth_r.x)/2 * w, (mouth_l.y + mouth_r.y)/2 * h])

            # เช็คท่าลูบปาก
            mouth_threshold = shoulder_width * 0.30
            touching_mouth = (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or \
                             (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold) or \
                             (np.linalg.norm(r_index_px - mouth_px) < mouth_threshold) or \
                             (np.linalg.norm(l_index_px - mouth_px) < mouth_threshold)

            # เช็คท่าค้ำศอก
            support_threshold = shoulder_width * 0.25
            supporting_elbow = (np.linalg.norm(r_wrist_px - l_elbow_px) < support_threshold) or \
                               (np.linalg.norm(r_index_px - l_elbow_px) < support_threshold) or \
                               (np.linalg.norm(l_wrist_px - r_elbow_px) < support_threshold) or \
                               (np.linalg.norm(l_index_px - r_elbow_px) < support_threshold)

            is_thinking_pose = touching_mouth or supporting_elbow

            # เช็คกอดอก 
            r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle_ok = 60 <= r_arm_angle <= 115
            l_angle_ok = 60 <= l_arm_angle <= 115
            
            arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
            is_arms_crossed = r_angle_ok and l_angle_ok and arms_intersect

            # ระบบตรวจจับโซน 
            def get_touched_zone(wrist, elbow, shoulder, index_finger, ear):
                if wrist.visibility < 0.5 and index_finger.visibility < 0.5: return None
                
                arm_angle = calculate_angle(shoulder, elbow, wrist)
                if arm_angle >= 115: return None 
                
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                index_px = (int(index_finger.x * w), int(index_finger.y * h))
                
                def is_in_poly(poly):
                    return (cv2.pointPolygonTest(poly, wrist_px, False) >= 0) or \
                           (cv2.pointPolygonTest(poly, index_px, False) >= 0)

                if is_in_poly(head_pts): 
                    if index_finger.z > (ear.z + 0.02):
                        scratch_flag[0] = True 
                        return None
                    return "HEAD"
                
                if is_in_poly(chest_pts): return "CHEST"
                if is_in_poly(abd_pts): return "ABDOMEN"
                
                return None

            r_zone = get_touched_zone(r_wrist, r_elbow, r_shoulder, r_index, r_ear)
            l_zone = get_touched_zone(l_wrist, l_elbow, l_shoulder, l_index, l_ear)

            # อัปเดตสถานะ
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

            # วาดหลอด Loading Bar (คงไว้ตำแหน่งเดิมให้เห็นชัด)
            if active_zone:
                bar_width = int((duration / CONFIRMATION_TIME) * 200)
                if bar_width > 200: bar_width = 200 
                cv2.rectangle(image, (50, 400), (50 + bar_width, 430), (0, 255, 255), -1)
                cv2.rectangle(image, (50, 400), (250, 430), (255, 255, 255), 2)
                cv2.putText(image, f"Analyzing {active_zone}... {duration:.1f}s", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ---------------------------------------------------------
            # NEW UI: AI TRIAGE MONITOR DASHBOARD (มุมขวาบน)
            # ---------------------------------------------------------
            panel_w, panel_h = 350, 140
            panel_x, panel_y = w - panel_w - 20, 20

            # สร้างพื้นหลังโปร่งแสง
            overlay = image.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # ส่วนหัว Dashboard
            cv2.putText(image, "Monitor", (panel_x + 15, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.line(image, (panel_x + 15, panel_y + 45), (panel_x + panel_w - 15, panel_y + 45), (100, 100, 100), 2)

            y_offset = panel_y + 75
            
            # ลอจิกการแสดงข้อความบน Dashboard
            if is_arms_crossed:
                cv2.putText(image, "State : IGNORED", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(image, "Reason: Crossed Arms", (panel_x + 15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif is_thinking_pose:
                cv2.putText(image, "State : IGNORED", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(image, "Reason: Thinking Pose", (panel_x + 15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif scratch_flag[0]:
                cv2.putText(image, "State : IGNORED", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(image, "Reason: Scratching Head", (panel_x + 15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                if duration >= CONFIRMATION_TIME:
                    # ใส่ไฟกระพริบเมื่อ Confirmed
                    pulse = int(abs(math.sin(time.time() * 6)) * 255)
                    cv2.circle(image, (panel_x + panel_w - 30, panel_y + 25), 8, (0, 0, pulse), -1)
                    
                    cv2.putText(image, "State : ALERT CONFIRMED!", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(image, f"Detect: {active_zone} Significant Condition", (panel_x + 15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif active_zone:
                    cv2.putText(image, "State : ANALYZING...", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(image, f"Target: {active_zone}", (panel_x + 15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(image, "State : STANDBY", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image, "Target: Normal Posture", (panel_x + 15, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            cv2.putText(image, f"Right Ang: {int(r_arm_angle)}",

                            (int(r_elbow.x * w), int(r_elbow.y * h)),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            cv2.putText(image, f"Left Ang: {int(l_arm_angle)}",

                            (int(l_elbow.x * w), int(l_elbow.y * h)),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Smart Triage V9.0', image)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()