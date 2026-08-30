# import cv2
# import mediapipe as mp
# import math
# import time 
# import numpy as np
# import json
# import requests # สำหรับเรียก Ollama API หรือ Local LLM

# # --- CONFIGURATION ---
# LLM_MODEL = "llama3.2:3b" # ชื่อโมเดลที่คุณโหลดไว้ในเครื่อง
# ESI_PROMPT_SYSTEM = """
# You are a Medical Triage Assistant. Analyze the provided Visual Flags and Patient Text.
# Assign an ESI Level (1-5) based on standard triage rules which is
# ●	Level 1 (Resuscitation)
# •	Definition: A patient with a life threatening condition, that needs help right away. Without immediate medical treatment, the patient is at high risk of dying.
# •	Symptom: Unconscious, have stopped breathing or are gasping for air or are experiencing heart failure or severe shock.
# ●	Level 2 (Emergent)
# •	Definition: Is in a high risk situation, experiencing severe pain (between 7 - 10), or showing with acute confusion. Treatment must be start within 10 - 15 minutes.
# •	Symptom: Severe chest pain, unilateral weakness, suspected stroke or severe dyspnea.
# ●	Level 3 (Urgent)
# •	Definition: Has stable vital signs but requires "two or more evidence" for evaluation and treatment. Treatment should begin within 30-60 minutes.
# •	Symptom: Severe abdominal pain and closed fractures requiring X-rays and splinting or high fever with dyspnea, etc.
# ●	Level 4 (Less Urgent)
# •	Definition: Has stable vital signs and is expected to require "only one evidence" for evaluation and treatment.
# •	Symptom: Minor lacerations, sprained ankle, or suspected urinary tract infection (UTI).
# ●	Level 5 (non urgent)
# •	Definition: General patients seeking treatment for minor illnesses. Require no resources beyond a standard physical examination and a medical prescription.
# •	Symptom: Common cold, sore throat, minor rashes, prescription refills or routine wound dressing.
# .
# Output ONLY a valid JSON with keys: "esi_level" and "clinical_summary".
# Do NOT diagnose a specific disease.
# """

# # --- LLM FUNCTION ---
# def call_llama_reasoning(visual_flag, skin_status, patient_text):
#     print("\n--- Sending Data to Llama 3.2 ---")
#     combined_input = f"Visual Posture: {visual_flag}\nSkin Status: {skin_status}\nPatient says: {patient_text}"
    
#     # ตัวอย่างการเรียกผ่าน Ollama API (รันในเครื่องตัวเอง)
#     payload = {
#         "model": LLM_MODEL,
#         "system": ESI_PROMPT_SYSTEM,
#         "prompt": combined_input,
#         "stream": False,
#         "format": "json"
#     }
    
#     try:
#         # สมมติว่ารัน Ollama ที่ port 11434
#         response = requests.post("http://localhost:11434/api/generate", json=payload)
#         return response.json()['response']
#     except Exception as e:
#         return f"Error connecting to LLM: {e}"

# # --- VISION INITIALIZATION ---
# mp_pose = mp.solutions.pose
# mp_face_mesh = mp.solutions.face_mesh
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# # (ฟังก์ชันคณิตศาสตร์ ccw, check_intersection, calculate_angle เหมือนเดิม)
# # ... [ข้ามไปส่วน Main Loop เพื่อความกระชับ] ...

# cap = cv2.VideoCapture(0)
# calibration_start = time.time()
# calibrated = False
# baseline_A = 0
# active_zone = None
# holding_start = 3

# print("System Starting... Please look at the camera for Skin Calibration.")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: break
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
    
#     # 1. ประมวลผล Vision (Pose & Face)
#     results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     results_face = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     current_visual_flag = "Normal"
#     current_skin_status = "NORMAL"
    
#     # --- SKIN LAB ANALYSIS (Simplified for Integration) ---
#     if results_face.multi_face_landmarks:
#         # (ตรรกะเจาะหน้ากาก LAB ที่เราคุยกัน เพื่อหาค่า avg_A)
#         # [จำลองค่า avg_A] -> ในการรันจริงให้ยกโค้ด LAB มาใส่ตรงนี้
#         avg_A = 135 # ค่าสมมติ
        
#         if not calibrated and (time.time() - calibration_start > 3):
#             baseline_A = avg_A
#             calibrated = True
#             print(f"Calibration Done. Baseline A: {baseline_A}")
        
#         if calibrated:
#             a_shift = avg_A - baseline_A
#             if a_shift > 1.5: current_skin_status = "FLUSHING (RED)"
#             elif a_shift < -1.5: current_skin_status = "PALLOR (PALE)"

#     # --- POSTURE DETECTION ---
#     # (ตรรกะกุมอก/กุมท้อง/กอดอก จาก V10.0)
#     # สมมติผลลัพธ์ว่าอยู่ใน active_zone และผ่านการเช็ค Edge Case แล้ว
#     # หากตรวจพบ Alert Confirmed:
    
#     if active_zone and (time.time() - holding_start > 2.0):
#         # !! TRIGGER POINT !!
#         print(f"\n[ALERT] System detected: {active_zone} and Skin: {current_skin_status}")
#         cv2.putText(frame, ">>> SYSTEM TRIGGERED: GO TO TERMINAL <<<", (50, h//2), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#         cv2.imshow('Smart Triage Integration', frame)
#         cv2.waitKey(1)
        
#         # รับค่า Text แทนเสียงพูด
#         patient_voice_text = input("Patient says (Thai/Eng): ")
        
#         # ส่งเข้า LLM
#         json_result = call_llama_reasoning(active_zone, current_skin_status, patient_voice_text)
        
#         print("\n=== FINAL TRIAGE RESULT (JSON) ===")
#         print(json_result)
#         print("===================================\n")
        
#         # Reset เพื่อรับเคสใหม่
#         active_zone = None
#         holding_start = 0
#         time.sleep(2) # หยุดรอแป๊บนึง

#     # UI Dashboard (วาด Overlay)
#     # ... [วาดผลลัพธ์บนหน้าจอเหมือน V10.0] ...
    
#     cv2.imshow('Smart Triage Integration', frame)
#     if cv2.waitKey(5) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()










# import cv2
# import mediapipe as mp
# import math
# import time 
# import numpy as np
# import requests

# # --- CONFIGURATION สำหรับ LLM ---
# LLM_MODEL = "llama3.2:3b"
# ESI_PROMPT_SYSTEM = """
# You are a Medical Triage Assistant. Analyze the provided Visual Flags and Patient Text.
# Assign an ESI Level (1-5) based on standard triage rules:
# - Level 1: Life threatening, unconscious, severe shock.
# - Level 2: High risk, severe pain (7-10), chest pain, suspected stroke.
# - Level 3: Stable vitals but requires 2+ resources (e.g. severe abdominal pain).
# - Level 4: Stable vitals, requires 1 resource.
# - Level 5: Non-urgent, common cold, minor issues.
# Output ONLY a valid JSON with keys: "esi_level" and "clinical_summary".
# Do NOT diagnose a specific disease.
# """

# def call_llama_reasoning(visual_flag, skin_status, patient_text):
#     print("\n[LLM] --- Sending Data to Llama 3.2 ---")
#     combined_input = f"Visual Posture: {visual_flag}\nSkin Status: {skin_status}\nPatient says: {patient_text}"
#     payload = {
#         "model": LLM_MODEL,
#         "system": ESI_PROMPT_SYSTEM,
#         "prompt": combined_input,
#         "stream": False,
#         "format": "json"
#     }
#     try:
#         response = requests.post("http://localhost:11434/api/generate", json=payload)
#         return response.json()['response']
#     except Exception as e:
#         return f'{{"error": "Error connecting to LLM: {e}"}}'

# # --- VISION ENGINE SETUP ---
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def calculate_angle(a, b, c):
#     a = [a.x, a.y]; b = [b.x, b.y]; c = [c.x, c.y] 
#     radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
#     angle = abs(radians*180.0/math.pi)
#     if angle > 180.0: angle = 360-angle
#     return angle

# def ccw(A, B, C): return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
# def check_intersection(A, B, C, D): return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# cap = cv2.VideoCapture(0)

# # --- ตัวแปรควบคุม ---
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
# current_skin_status = "Normal"

# print(">>> System Ready: Running Vision Engine...")

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
#             mouth_l = landmarks[9]; mouth_r = landmarks[10]
#             r_eye = landmarks[5]; l_eye = landmarks[2]
#             r_ear = landmarks[8]; l_ear = landmarks[7]
#             r_shoulder = landmarks[12]; l_shoulder = landmarks[11]
#             r_hip = landmarks[24]; l_hip = landmarks[23] 
#             r_elbow = landmarks[14]; l_elbow = landmarks[13]
#             r_wrist = landmarks[16]; l_wrist = landmarks[15]
#             r_index = landmarks[20]; l_index = landmarks[19]

#             # --- 1. SKIN ANALYSIS (HSV Baseline ตัวเดิมไปก่อน) ---
#             if skin_calibrating and skin_calibration_start == 0:
#                 skin_calibration_start = time.time()

#             r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
#             r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
#             box_size = 15
            
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
#                 else:
#                     s_drop = baseline_S - avg_s 
#                     h_shift = baseline_H - avg_h 
#                     if s_drop > 25: current_skin_status = "Pallor (Pale)"
#                     elif h_shift > 10 and avg_s > baseline_S + 10: current_skin_status = "Flushing (RED)"
#                     else: current_skin_status = "Normal"

#             # --- 2. GESTURE DETECTION (ตรรกะคณิตศาสตร์) ---
#             r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
#             l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
#             r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
#             l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
            
#             chest_ratio = 0.55
#             r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
#             l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio
#             chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
#             abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))

#             nose_px = np.array([nose.x * w, nose.y * h])
#             shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
#             base_head_h = shoulder_width * 0.58
#             head_tl = nose_px + np.array([-shoulder_width*0.3, -base_head_h])
#             head_br = nose_px + np.array([shoulder_width*0.3, base_head_h*0.15])
#             head_pts = np.array([head_tl, [head_br[0], head_tl[1]], head_br, [head_tl[0], head_br[1]]], np.int32).reshape((-1, 1, 2))

#             # Edge Cases
#             r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h]); r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
#             r_index_px = np.array([r_index.x * w, r_index.y * h]) 
#             l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h]); l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
#             l_index_px = np.array([l_index.x * w, l_index.y * h]) 
#             mouth_px = np.array([(mouth_l.x + mouth_r.x)/2 * w, (mouth_l.y + mouth_r.y)/2 * h])

#             mouth_threshold = shoulder_width * 0.30
#             is_thinking_pose = (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold)

#             r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
#             l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
#             arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
#             is_arms_crossed = (60 <= r_arm_angle <= 115) and (60 <= l_arm_angle <= 115) and arms_intersect

#             def get_touched_zone(wrist, index_finger):
#                 wrist_px = (int(wrist.x * w), int(wrist.y * h))
#                 index_px = (int(index_finger.x * w), int(index_finger.y * h))
#                 def is_in_poly(poly):
#                     return (cv2.pointPolygonTest(poly, wrist_px, False) >= 0) or (cv2.pointPolygonTest(poly, index_px, False) >= 0)
#                 if is_in_poly(head_pts): return "Head"
#                 if is_in_poly(chest_pts): return "Chest"
#                 if is_in_poly(abd_pts): return "Abdomen"
#                 return None

#             r_zone = get_touched_zone(r_wrist, r_index)
#             l_zone = get_touched_zone(l_wrist, l_index)

#             if is_arms_crossed or is_thinking_pose: current_zone = None
#             else: current_zone = r_zone if r_zone else l_zone

#             if current_zone:
#                 if current_zone != active_zone:
#                     active_zone = current_zone
#                     holding_start_time = time.time()
#                 duration = time.time() - holding_start_time
#             else:
#                 active_zone = None
#                 holding_start_time = 0
#                 duration = 0

#             # --- 3. UI DASHBOARD ---
#             panel_w, panel_h = 350, 150
#             panel_x, panel_y = w - panel_w - 20, 20
#             overlay = image.copy()
#             cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
#             cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

#             cv2.putText(image, "Debug", (panel_x + 15, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             y_offset = panel_y + 70

#             if is_arms_crossed: cv2.putText(image, "Posture: IGNORED (Crossed)", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
#             elif active_zone: cv2.putText(image, f"ANALYZING {active_zone}", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#             else: cv2.putText(image, "Posture: Normal", (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 15, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             # --- 4. INTEGRATION TRIGGER (The Brain) ---
#             if duration >= CONFIRMATION_TIME:
#                 # วาดข้อความแจ้งเตือนสีแดงกลางจอ แล้วอัปเดตภาพ 1 ครั้งก่อนหยุด
#                 # cv2.putText(image, "Anomaly Detected", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#                 cv2.imshow('Smart Triage V10 - LLM Integrated', image)
#                 cv2.waitKey(1) 
                
#                 print(f"\n[TRIGGER] Patient is holding {active_zone} for 2s.")
#                 patient_voice = input("ENTER PATIENT SYMPTOM (Thai/Eng): ")
                
#                 # เรียก Llama 3.2
#                 json_result = call_llama_reasoning(active_zone, current_skin_status, patient_voice)
                
#                 print("\n" + "."*40)
#                 print("FINAL TRIAGE RESULT (JSON)")
#                 print("."*40)
#                 print(json_result)
#                 print("."*40 + "\n")
                
              
#                 active_zone = None
#                 holding_start_time = 0
#                 time.sleep(1)

#         cv2.imshow('Smart Triage V10 - LLM Part', image)
#         if cv2.waitKey(5) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()






























import cv2
import mediapipe as mp
import math
import time 
import numpy as np
import requests

# --- CONFIGURATION สำหรับ LLM ---
LLM_MODEL = "llama3.2:latest"
ESI_PROMPT_SYSTEM = """
You are a Medical Triage Assistant. Analyze the provided Visual Flags and Patient Text.
Assign an ESI Level (1-5) based on standard triage rules:
- Level 1: Life threatening, unconscious, severe shock.
- Level 2: High risk, severe pain (7-10), chest pain, suspected stroke.
- Level 3: Stable vitals but requires 2+ resources (e.g. severe abdominal pain).
- Level 4: Stable vitals, requires 1 resource.
- Level 5: Non-urgent, common cold, minor issues.
Output ONLY a valid JSON with keys: "esi_level" and "clinical_summary".
Do NOT diagnose a specific disease.
"""

# def call_llama_reasoning(visual_flag, skin_status, patient_text):
#     print("\nSending Data to Llama 3.2")
#     combined_input = f"Visual Posture: {visual_flag}\nSkin Status: {skin_status}\nPatient says: {patient_text}"
#     payload = {
#         "model": LLM_MODEL,  # เช็คชื่อโมเดลให้ตรงกับในเครื่องด้วยนะ!
#         "system": ESI_PROMPT_SYSTEM,
#         "prompt": combined_input,
#         "stream": False,
#         "format": "json" # ถ้า Error อีก ให้ลองลบบรรทัดนี้ทิ้งครับ
#     }
#     try:
#         response = requests.post("http://localhost:11434/api/generate", json=payload)
#         data = response.json()
        
#         # ดัก Error จาก Ollama
#         if 'response' in data:
#             return data['response']
#         else:
#             return f"\n[Ollama API Error Details]: {data}\n" # จะบอกเลยว่าเกิดอะไรขึ้น
            
#     except Exception as e:
#         return f'{{"error": "Error connecting to LLM: {e}"}}'




def call_llama_reasoning(visual_flag, skin_status, patient_text):
    print("\nSending Data to Llama 3.2 (4-bit)")
    combined_input = f"Visual Posture: {visual_flag}\nSkin Status: {skin_status}\nPatient says: {patient_text}"
    payload = {
        "model": LLM_MODEL, 
        "system": ESI_PROMPT_SYSTEM,
        "prompt": combined_input,
        "stream": False,
        "format": "json"
    }
    
    try:
        # 1. เริ่มจับเวลาตรงนี้!
        start_time = time.time() 
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        
        # 2. หยุดจับเวลาทันทีที่ได้คำตอบ!
        end_time = time.time() 
        
        latency = end_time - start_time # คำนวณเวลาที่ใช้ไป
        
        data = response.json()
        
        # 3. ปริ้นท์โชว์ใน Terminal แบบเท่ๆ
        print("="*50)
        print(f"Optimization Successful.")
        print(f"Inference Time: {latency:.2f} seconds") 
        # print("="*50)

        if 'response' in data:
            return data['response']
        else:
            return f"\n[Ollama API Error]: {data}\n"
            
    except Exception as e:
        return f'{{"error": "Error: {e}"}}'


# --- VISION ENGINE SETUP ---
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = [a.x, a.y]; b = [b.x, b.y]; c = [c.x, c.y] 
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def ccw(A, B, C): return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def check_intersection(A, B, C, D): return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

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
current_skin_status = "Normal"

print("System Ready.")

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
            mouth_l = landmarks[9]; mouth_r = landmarks[10]
            r_eye = landmarks[5]; l_eye = landmarks[2]
            r_ear = landmarks[8]; l_ear = landmarks[7]
            r_shoulder = landmarks[12]; l_shoulder = landmarks[11]
            r_hip = landmarks[24]; l_hip = landmarks[23] 
            r_elbow = landmarks[14]; l_elbow = landmarks[13]
            r_wrist = landmarks[16]; l_wrist = landmarks[15]
            r_index = landmarks[20]; l_index = landmarks[19]

            # --- 1. SKIN ANALYSIS ---
            if skin_calibrating and skin_calibration_start == 0:
                skin_calibration_start = time.time()

            r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
            r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
            box_size = 15

            # --- cheek ROI boxes HIDDEN ---
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
                else:
                    s_drop = baseline_S - avg_s 
                    h_shift = baseline_H - avg_h 
                    if s_drop > 25: current_skin_status = "Pallor (Pale)"
                    elif h_shift > 10 and avg_s > baseline_S + 10: current_skin_status = "Flushing (RED)"
                    else: current_skin_status = "Normal"

            # --- 2. GESTURE DETECTION ---
            r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
            l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
            r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
            l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
            
            chest_ratio = 0.55
            r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
            l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio

            # --- zone polygons computed but NOT drawn ---
            chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
            abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))

            nose_px = np.array([nose.x * w, nose.y * h])
            shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
            base_head_h = shoulder_width * 0.58
            head_tl = nose_px + np.array([-shoulder_width*0.3, -base_head_h])
            head_br = nose_px + np.array([shoulder_width*0.3, base_head_h*0.15])
            head_pts = np.array([head_tl, [head_br[0], head_tl[1]], head_br, [head_tl[0], head_br[1]]], np.int32).reshape((-1, 1, 2))
            # --- head box NOT drawn ---

            r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h])
            r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
            r_index_px = np.array([r_index.x * w, r_index.y * h]) 
            l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h])
            l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
            l_index_px = np.array([l_index.x * w, l_index.y * h]) 
            mouth_px = np.array([(mouth_l.x + mouth_r.x)/2 * w, (mouth_l.y + mouth_r.y)/2 * h])

            mouth_threshold = shoulder_width * 0.30
            is_thinking_pose = (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or \
                               (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold)

            r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
            is_arms_crossed = (60 <= r_arm_angle <= 115) and (60 <= l_arm_angle <= 115) and arms_intersect

            def get_touched_zone(wrist, index_finger):
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                index_px = (int(index_finger.x * w), int(index_finger.y * h))
                def is_in_poly(poly):
                    return (cv2.pointPolygonTest(poly, wrist_px, False) >= 0) or \
                           (cv2.pointPolygonTest(poly, index_px, False) >= 0)
                if is_in_poly(head_pts): return "Head"
                if is_in_poly(chest_pts): return "Chest"
                if is_in_poly(abd_pts): return "Abdomen"
                return None

            r_zone = get_touched_zone(r_wrist, r_index)
            l_zone = get_touched_zone(l_wrist, l_index)

            if is_arms_crossed or is_thinking_pose: current_zone = None
            else: current_zone = r_zone if r_zone else l_zone

            if current_zone:
                if current_zone != active_zone:
                    active_zone = current_zone
                    holding_start_time = time.time()
                duration = time.time() - holding_start_time
            else:
                active_zone = None
                holding_start_time = 0
                duration = 0

            # --- 3. UI DASHBOARD ONLY ---
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
                cv2.putText(image, "Posture: IGNORED (Crossed)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
            elif is_thinking_pose:
                cv2.putText(image, "Posture: IGNORED (Thinking)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
            elif duration >= CONFIRMATION_TIME:
                pulse = int(abs(math.sin(time.time() * 6)) * 255)
                cv2.circle(image, (panel_x + panel_w - 22, panel_y + 18), 6, (0, 0, pulse), -1)
                cv2.putText(image, f"Posture: ALERT ({active_zone})", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
            elif active_zone:
                cv2.putText(image, f"Posture: ANALYZING {active_zone}...", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
            else:
                cv2.putText(image, "Posture: Normal", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

            y_skin = y_offset + 28
            if skin_calibrating:
                cal_time_left = max(0, CALIBRATION_TIME - (time.time() - skin_calibration_start))
                cv2.putText(image, f"Skin   : CALIBRATING ({cal_time_left:.1f}s)", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
            elif current_skin_status == "Pallor (Pale)":
                cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            elif current_skin_status == "Flushing (RED)":
                cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
            else:
                cv2.putText(image, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
            

            # --- skeleton/landmarks HIDDEN (mp_drawing.draw_landmarks removed) ---

            # --- 4. LLM TRIGGER ---
            if duration >= CONFIRMATION_TIME:
                cv2.imshow('Smart Triage V10- Result', image)
                cv2.waitKey(1) 
                
                # print(f"\nPatient is holding {active_zone} for 2s.")
                patient_voice = input("PATIENT SYMPTOM (Thai/Eng): ")
                
                json_result = call_llama_reasoning(active_zone, current_skin_status, patient_voice)
                
                print("\n")
                # print("FINAL TRIAGE RESULT (JSON)")
                # print("."*40)
                print(json_result)
                # print("."*40 + "\n")

                active_zone = None
                holding_start_time = 0
                time.sleep(1)

        cv2.imshow('Smart Triage V10 - LLM Part', image)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


