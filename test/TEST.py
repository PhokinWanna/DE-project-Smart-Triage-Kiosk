import cv2
import mediapipe as mp
import math
import time 
import numpy as np
import requests

LLM_MODEL = "llama3.2:latest"

ESI_PROMPT_SYSTEM = """
You are a Medical Triage Assistant. Analyze the Visual Posture Flag, Skin Status, and Patient verbal report.
Assign an ESI Level (1-5):
- Level 1: Life threatening, unconscious, severe shock.
- Level 2: High risk, severe pain (7-10), chest pain, suspected stroke.
- Level 3: Stable vitals, requires 2+ resources (e.g. head pain, abdominal pain).
- Level 4: Stable vitals, requires 1 resource.
- Level 5: Non-urgent, minor issues.

VISUAL FLAG RULES:
The "Visual Posture Flag" is an objective sensor reading — the patient physically held or guarded that body zone.
Use it as clinical evidence alongside the verbal report.

You will receive a SCENARIO tag in the input. Follow it strictly:

SCENARIO A — CONFIRMED:
  Visual flag and verbal report are aligned. Patient is reporting symptoms matching the detected zone.
  → Assign ESI based on combined evidence.
  → Write a clean, direct clinical summary of the symptoms.
  → Do NOT write any language about denial, conflict, or hesitation. There is none.

SCENARIO B — CONFLICT:
  Visual flag is present but verbal report does not match or denies symptoms.
  → Visual evidence overrides verbal denial.
  → Assign ESI based on the visual flag zone minimum.
  → Note in the summary that verbal report did not match the observed guarding behavior.

SCENARIO C — NO FLAG:
  No visual flag detected. Base ESI entirely on verbal report and skin status.

Zone minimum ESI when a flag is present:
  Head     → minimum ESI 3
  Chest    → minimum ESI 2
  Abdomen  → minimum ESI 3

Only assign ESI 5 when there is NO visual flag AND verbal report is genuinely non-urgent.

Output ONLY valid JSON with keys: "esi_level" (integer 1-5) and "clinical_summary" (string).
Do NOT diagnose a specific disease.
"""

ZONE_CONFIRM_KEYWORDS = {
    "Head": [
        "head", "หัว", "skull", "migraine", "ไมเกรน", "dizzy", "วิงเวียน",
        "headache", "ปวดหัว", "forehead", "temple", "neck", "คอ",
        "blurry", "ตามัว", "nausea", "คลื่นไส้", "faint", "เป็นลม"
    ],
    "Chest": [
        "chest", "หน้าอก", "heart", "หัวใจ", "breath", "หายใจ",
        "tight", "แน่น", "pressure", "palpitation", "ใจสั่น",
        "lung", "ปอด", "cough", "ไอ", "shortness", "เหนื่อย"
    ],
    "Abdomen": [
        "stomach", "ท้อง", "abdomen", "belly", "nausea", "คลื่นไส้",
        "vomit", "อาเจียน", "cramp", "ปวดท้อง", "bowel", "ลำไส้",
        "bloat", "แน่นท้อง", "diarrhea", "ท้องเสีย"
    ]
}

def detect_scenario(visual_flag, patient_text):
    """
    Returns 'CONFIRMED', 'CONFLICT', or 'NO_FLAG'.
    CONFIRMED  = visual flag present AND patient words match the zone.
    CONFLICT   = visual flag present BUT patient words do not match.
    NO_FLAG    = no visual flag at all.
    """
    if not visual_flag or visual_flag == "None":
        return "NO_FLAG"

    text_lower = patient_text.lower()
    keywords = ZONE_CONFIRM_KEYWORDS.get(visual_flag, [])
    for kw in keywords:
        if kw.lower() in text_lower:
            return "CONFIRMED"
    return "CONFLICT"


def call_llama_reasoning(visual_flag, skin_status, patient_text):
    print("\nSending Data to Llama 3.2 (4-bit)")

    scenario = detect_scenario(visual_flag, patient_text)

    if scenario == "CONFIRMED":
        scenario_note = (
            "SCENARIO: A — CONFIRMED\n"
            f"The patient verbally reported symptoms that match the visual flag zone '{visual_flag}'.\n"
            "Both sources agree. Write a clean, direct clinical summary. "
            "Do NOT mention denial, conflict, or hesitation — there is none."
        )
    elif scenario == "CONFLICT":
        scenario_note = (
            "SCENARIO: B — CONFLICT\n"
            f"Visual sensor detected sustained guarding of '{visual_flag}', "
            "but the patient's verbal report does not mention related symptoms.\n"
            "Override the verbal report. Assign ESI based on the zone. "
            "Note in the summary that the verbal report did not match the observed behavior."
        )
    else:
        scenario_note = (
            "SCENARIO: C — NO FLAG\n"
            "No posture flag was detected. Base ESI entirely on verbal report and skin status."
        )

    combined_input = (
        f"Visual Posture Flag (objective): {visual_flag if visual_flag else 'None'}\n"
        f"Skin Status: {skin_status}\n"
        f"Patient verbal report: {patient_text}\n\n"
        f"{scenario_note}"
    )

    payload = {
        "model": LLM_MODEL,
        "system": ESI_PROMPT_SYSTEM,
        "prompt": combined_input,
        "stream": False,
        "format": "json"
    }

    try:
        start_time = time.time()
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        end_time = time.time()
        latency = end_time - start_time
        data = response.json()

        print("="*50)
        print(f"Optimization Successful.")
        print(f"Inference Time : {latency:.2f} seconds")
        print(f"Scenario       : {scenario}")

        if 'response' in data:
            return data['response']
        else:
            return f"\n[Ollama API Error]: {data}\n"

    except Exception as e:
        return f'{{"error": "Error: {e}"}}'


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = [a.x, a.y]; b = [b.x, b.y]; c = [c.x, c.y]
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def ccw(A, B, C): return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def check_intersection(A, B, C, D): return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

cap = cv2.VideoCapture(1)

holding_start_time = 0
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
clean_image = None

print("System Ready. Please step in front of the camera.")

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

            if skin_calibrating and skin_calibration_start == 0:
                skin_calibration_start = time.time()

            r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
            r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
            box_size = 15

            r_cheek_roi = image[max(0, r_cheek_y-box_size):min(h, r_cheek_y+box_size),
                                max(0, r_cheek_x-box_size):min(w, r_cheek_x+box_size)]

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

            r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
            l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
            r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
            l_hip_px = np.array([l_hip.x * w, l_hip.y * h])

            chest_ratio = 0.55
            r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * chest_ratio
            l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * chest_ratio

            chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
            abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))

            nose_px = np.array([nose.x * w, nose.y * h])
            shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
            base_head_h = shoulder_width * 0.58
            head_tl = nose_px + np.array([-shoulder_width*0.3, -base_head_h])
            head_br = nose_px + np.array([shoulder_width*0.3, base_head_h*0.15])
            head_pts = np.array([head_tl, [head_br[0], head_tl[1]], head_br, [head_tl[0], head_br[1]]], np.int32).reshape((-1, 1, 2))

            cv2.polylines(image, [head_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(image, [chest_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(image, [abd_pts], isClosed=True, color=(0, 165, 255), thickness=2)

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
            arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
            is_arms_crossed = (60 <= r_arm_angle <= 115) and (60 <= l_arm_angle <= 115) and arms_intersect

            def get_touched_zone(wrist, index_finger, ear):
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

            r_zone = get_touched_zone(r_wrist, r_index, r_ear)
            l_zone = get_touched_zone(l_wrist, l_index, l_ear)

            if is_arms_crossed or is_thinking_pose or scratch_flag[0]:
                current_zone = None
            else:
                current_zone = r_zone if r_zone else l_zone

            if current_zone:
                if current_zone != active_zone:
                    active_zone = current_zone
                    holding_start_time = time.time()
                duration = time.time() - holding_start_time
            else:
                active_zone = None
                holding_start_time = 0
                duration = 0

            clean_image = image.copy()

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for img_to_draw in [clean_image, image]:
                panel_w, panel_h = 290, 120
                panel_x, panel_y = w - panel_w - 15, 15
                overlay = img_to_draw.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, img_to_draw, 0.3, 0, img_to_draw)

                cv2.putText(img_to_draw, "Debug", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.line(img_to_draw, (panel_x + 10, panel_y + 32), (panel_x + panel_w - 10, panel_y + 32), (100, 100, 100), 1)

                y_offset = panel_y + 55
                font_scale = 0.48

                if is_arms_crossed:
                    cv2.putText(img_to_draw, "Posture: IGNORED (Crossed)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
                elif is_thinking_pose:
                    cv2.putText(img_to_draw, "Posture: IGNORED (Thinking)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
                elif scratch_flag[0]:
                    cv2.putText(img_to_draw, "Posture: IGNORED (Scratch)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
                elif duration >= CONFIRMATION_TIME:
                    pulse = int(abs(math.sin(time.time() * 6)) * 255)
                    cv2.circle(img_to_draw, (panel_x + panel_w - 22, panel_y + 18), 6, (0, 0, pulse), -1)
                    cv2.putText(img_to_draw, f"Posture: ALERT ({active_zone})", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
                elif active_zone:
                    cv2.putText(img_to_draw, f"Posture: ANALYZING {active_zone}...", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
                else:
                    cv2.putText(img_to_draw, "Posture: Normal", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

                y_skin = y_offset + 28
                if skin_calibrating:
                    cal_time_left = max(0, CALIBRATION_TIME - (time.time() - skin_calibration_start))
                    cv2.putText(img_to_draw, f"Skin   : CALIBRATING ({cal_time_left:.1f}s)", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
                elif current_skin_status == "Pallor (Pale)":
                    cv2.putText(img_to_draw, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                elif current_skin_status == "Flushing (RED)":
                    cv2.putText(img_to_draw, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
                else:
                    cv2.putText(img_to_draw, f"Skin   : {current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

            if duration >= CONFIRMATION_TIME:
                cv2.imshow('Skeleton View', image)
                cv2.imshow('Clean View', clean_image)
                cv2.waitKey(1)

                patient_voice = input("PATIENT SYMPTOM (Thai/Eng): ")
                json_result = call_llama_reasoning(active_zone, current_skin_status, patient_voice)

                print("\n")
                print(json_result)

                active_zone = None
                holding_start_time = 0
                time.sleep(1)

        if clean_image is not None:
            cv2.imshow('Clean View', clean_image)
        cv2.imshow('Skeleton View', image)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()