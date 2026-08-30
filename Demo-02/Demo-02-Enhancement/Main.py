"""
Smart Triage Kiosk - Main Integration Module
Vision -> Reasoning -> Audio Pipeline

Improvements:
- Centralized configuration from config.py
- Structured state management
- Resource cleanup with try/finally
- Logging throughout
- Performance optimizations (removed nested function, single UI draw pass)
- Proper error handling
"""
import cv2
import mediapipe as mp
import math
import time
import numpy as np
import logging

from V_Module import calculate_angle, check_intersection
from R_Module import call_llama_reasoning
from A_Module import listen_to_patient, handle_output_guardrail, speak_audio,pygame
import config

if config.ENABLE_LOGGING:
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class TriageKioskState:
    """Encapsulate all state variables for better management."""
    def __init__(self):
        self.holding_start_time = 0
        self.active_zone = None
        self.skin_calibrating = True
        self.skin_calibration_start = 0
        self.baseline_A = 0.0
        self.A_history = []
        self.current_skin_status = "Normal"
        self.clean_image = None
        self.duration = 0
    
    def reset_zone(self):
        """Reset zone tracking after processing."""
        self.active_zone = None
        self.holding_start_time = 0
        self.duration = 0


def get_touched_zone(wrist, index_finger, ear, w, h, head_pts, chest_pts, abd_pts):
    """
    IMPROVED: Moved OUTSIDE the main loop to avoid redefinition every frame.
    Determine which body zone the hand is touching.
    """
    wrist_px = (int(wrist.x * w), int(wrist.y * h))
    index_px = (int(index_finger.x * w), int(index_finger.y * h))
    
    def is_in_poly(poly):
        return (cv2.pointPolygonTest(poly, wrist_px, False) >= 0) or \
               (cv2.pointPolygonTest(poly, index_px, False) >= 0)
    
    if is_in_poly(head_pts):
        if index_finger.z > (ear.z + config.HEAD_SCRATCH_Z_THRESHOLD):
            return None
        return "Head"
    if is_in_poly(chest_pts):
        return "Chest"
    if is_in_poly(abd_pts):
        return "Abdomen"
    return None


def draw_debug_panel(image, state, w, h, is_arms_crossed, is_thinking_pose, scratch_flag):
    """IMPROVED: Single draw pass instead of drawing twice."""
    panel_w = config.DEBUG_PANEL_WIDTH
    panel_h = config.DEBUG_PANEL_HEIGHT
    panel_x = w - panel_w - config.DEBUG_PANEL_OFFSET_X
    panel_y = config.DEBUG_PANEL_OFFSET_Y
    
    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    cv2.putText(image, "Debug", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    y_offset = panel_y + 55
    font_scale = config.DEBUG_FONT_SCALE
    
    if is_arms_crossed:
        cv2.putText(image, "Posture: IGNORED (Crossed)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
    elif is_thinking_pose:
        cv2.putText(image, "Posture: IGNORED (Thinking)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
    elif scratch_flag:
        cv2.putText(image, "Posture: IGNORED (Scratch)", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 1)
    elif state.duration >= config.CONFIRMATION_TIME:
        pulse = int(abs(math.sin(time.time() * 6)) * 255)
        cv2.circle(image, (panel_x + panel_w - 22, panel_y + 18), 6, (0, 0, pulse), -1)
        cv2.putText(image, f"Posture: ALERT ({state.active_zone})", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
    elif state.active_zone:
        cv2.putText(image, f"Posture: ANALYZING {state.active_zone}...", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
    else:
        cv2.putText(image, "Posture: Normal", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
    
    y_skin = y_offset + 28

    if state.skin_calibrating:
        cal_time_left = max(0, config.CALIBRATION_TIME - (time.time() - state.skin_calibration_start))
        cv2.putText(image, f"Skin   : CALIBRATING ({cal_time_left:.1f}s)", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
    else:
        color = (255, 255, 255) if state.current_skin_status == "Pallor (Pale)" else (0, 0, 255) if state.current_skin_status == "Flushing (RED)" else (0, 255, 0)
        cv2.putText(image, f"Skin   : {state.current_skin_status}", (panel_x + 10, y_skin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)


def main():
    """Main triage kiosk loop with improved error handling and resource management."""
    logger.info("=== Smart Triage Kiosk Started ===")
    
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        speak_audio("ระบบไม่สามารถเปิดกล้องได้ กรุณาติดต่อผู้ดูแล")
        return
    
    state = TriageKioskState()
    
    print("System Ready. Please step in front of the camera.")
    logger.info("System ready - waiting for patient")
    speak_audio("ระบบพร้อมทำงาน กรุณายืนหน้ากล้องค่ะ")
    
    try:
        with mp_pose.Pose(
            min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE
        ) as pose:
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                image = cv2.flip(image, 1)
                h, w, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image.flags.writeable = True
                
                scratch_flag = False
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    nose = landmarks[0]
                    mouth_l = landmarks[9]
                    mouth_r = landmarks[10]
                    l_eye = landmarks[2]
                    r_eye = landmarks[5]
                    l_ear = landmarks[7]
                    r_ear = landmarks[8]
                    l_shoulder = landmarks[11]
                    r_shoulder = landmarks[12]
                    l_elbow = landmarks[13]
                    r_elbow = landmarks[14]
                    l_wrist = landmarks[15]
                    r_wrist = landmarks[16]
                    l_index = landmarks[19]
                    r_index = landmarks[20]
                    l_hip = landmarks[23]
                    r_hip = landmarks[24]
                    
                    if state.skin_calibrating and state.skin_calibration_start == 0:
                        state.skin_calibration_start = time.time()
                        logger.info("Starting skin calibration")
                    
                    r_cheek_x = int((r_eye.x + nose.x + r_ear.x) / 3 * w)
                    r_cheek_y = int((r_eye.y + nose.y + r_ear.y) / 3 * h)
                    box_size = config.SKIN_ANALYSIS_ROI_SIZE
                    
                    r_cheek_roi = image[
                        max(0, r_cheek_y - box_size):min(h, r_cheek_y + box_size),
                        max(0, r_cheek_x - box_size):min(w, r_cheek_x + box_size)
                    ]
                    
                    if r_cheek_roi.size != 0:
                        lab_roi = cv2.cvtColor(r_cheek_roi, cv2.COLOR_BGR2LAB)
                        avg_a = np.mean(lab_roi[:, :, 1])
                        
                        if state.skin_calibrating:
                            state.A_history.append(avg_a)
                            
                            if time.time() - state.skin_calibration_start > config.CALIBRATION_TIME:
                                state.skin_calibrating = False
                                state.baseline_A = np.mean(state.A_history)
                                logger.info(f"Skin calibrated | Baseline A*: {state.baseline_A:.2f}")
                                print(f"[Skin Calibrated] Baseline A*: {state.baseline_A:.2f}")
                        else:
                            a_diff = avg_a - state.baseline_A
                            
                            if a_diff < config.SKIN_PALE_THRESHOLD:
                                state.current_skin_status = "Pallor (Pale)"
                            elif a_diff > config.SKIN_RED_THRESHOLD:
                                state.current_skin_status = "Flushing (RED)"
                            else:
                                state.current_skin_status = "Normal"
                    
                    r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
                    l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
                    r_hip_px = np.array([r_hip.x * w, r_hip.y * h])
                    l_hip_px = np.array([l_hip.x * w, l_hip.y * h])
                    
                    r_chest_bottom = r_sh_px + (r_hip_px - r_sh_px) * config.CHEST_RATIO_TO_BODY
                    l_chest_bottom = l_sh_px + (l_hip_px - l_sh_px) * config.CHEST_RATIO_TO_BODY
                    
                    chest_pts = np.array([r_sh_px, l_sh_px, l_chest_bottom, r_chest_bottom], np.int32).reshape((-1, 1, 2))
                    abd_pts = np.array([r_chest_bottom, l_chest_bottom, l_hip_px, r_hip_px], np.int32).reshape((-1, 1, 2))
                    
                    nose_px = np.array([nose.x * w, nose.y * h])
                    shoulder_width = np.linalg.norm(r_sh_px - l_sh_px)
                    base_head_h = shoulder_width * config.HEAD_HEIGHT_RATIO
                    head_tl = nose_px + np.array([-shoulder_width * config.HEAD_WIDTH_RATIO, -base_head_h])
                    head_br = nose_px + np.array([shoulder_width * config.HEAD_WIDTH_RATIO, base_head_h * 0.15])
                    head_pts = np.array([head_tl, [head_br[0], head_tl[1]], head_br, [head_tl[0], head_br[1]]], np.int32).reshape((-1, 1, 2))
                    
                    if config.SHOW_ZONE_POLYGONS:
                        cv2.polylines(image, [head_pts], isClosed=True, color=(0, 0, 255), thickness=2)
                        cv2.polylines(image, [chest_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        cv2.polylines(image, [abd_pts], isClosed=True, color=(0, 165, 255), thickness=2)
                    
                    r_elbow_px = np.array([r_elbow.x * w, r_elbow.y * h])
                    r_wrist_px = np.array([r_wrist.x * w, r_wrist.y * h])
                    r_index_px = np.array([r_index.x * w, r_index.y * h])
                    l_elbow_px = np.array([l_elbow.x * w, l_elbow.y * h])
                    l_wrist_px = np.array([l_wrist.x * w, l_wrist.y * h])
                    l_index_px = np.array([l_index.x * w, l_index.y * h])
                    mouth_px = np.array([(mouth_l.x + mouth_r.x) / 2 * w, (mouth_l.y + mouth_r.y) / 2 * h])
                    
                    mouth_threshold = shoulder_width * config.MOUTH_TOUCH_THRESHOLD_RATIO
                    touching_mouth = (
                        (np.linalg.norm(r_wrist_px - mouth_px) < mouth_threshold) or
                        (np.linalg.norm(l_wrist_px - mouth_px) < mouth_threshold) or
                        (np.linalg.norm(r_index_px - mouth_px) < mouth_threshold) or
                        (np.linalg.norm(l_index_px - mouth_px) < mouth_threshold)
                    )
                    
                    support_threshold = shoulder_width * config.ELBOW_SUPPORT_THRESHOLD_RATIO
                    supporting_elbow = (
                        (np.linalg.norm(r_wrist_px - l_elbow_px) < support_threshold) or
                        (np.linalg.norm(r_index_px - l_elbow_px) < support_threshold) or
                        (np.linalg.norm(l_wrist_px - r_elbow_px) < support_threshold) or
                        (np.linalg.norm(l_index_px - r_elbow_px) < support_threshold)
                    )
                    
                    is_thinking_pose = touching_mouth or supporting_elbow
                    
                    r_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    l_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    arms_intersect = check_intersection(r_elbow_px, r_index_px, l_elbow_px, l_index_px)
                    is_arms_crossed = (
                        (config.ARM_ANGLE_MIN <= r_arm_angle <= config.ARM_ANGLE_MAX) and
                        (config.ARM_ANGLE_MIN <= l_arm_angle <= config.ARM_ANGLE_MAX) and
                        arms_intersect
                    )
                    
                    r_zone = get_touched_zone(r_wrist, r_index, r_ear, w, h, head_pts, chest_pts, abd_pts)
                    l_zone = get_touched_zone(l_wrist, l_index, l_ear, w, h, head_pts, chest_pts, abd_pts)
                    
                    if r_zone == "Head" and r_index.z > (r_ear.z + config.HEAD_SCRATCH_Z_THRESHOLD):
                        scratch_flag = True
                    if l_zone == "Head" and l_index.z > (l_ear.z + config.HEAD_SCRATCH_Z_THRESHOLD):
                        scratch_flag = True
                    
                    if is_arms_crossed or is_thinking_pose or scratch_flag:
                        current_zone = None
                    else:
                        current_zone = r_zone if r_zone else l_zone
                    
                    if current_zone:
                        if current_zone != state.active_zone:
                            state.active_zone = current_zone
                            state.holding_start_time = time.time()
                            logger.info(f"Zone detected: {current_zone}")
                        state.duration = time.time() - state.holding_start_time
                    else:
                        state.reset_zone()
                    
                    state.clean_image = image.copy()
                    
                    if config.SHOW_SKELETON_LANDMARKS:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    if config.ENABLE_DEBUG_DISPLAY:
                        draw_debug_panel(image, state, w, h, is_arms_crossed, is_thinking_pose, scratch_flag)
                    
                    if state.duration >= config.CONFIRMATION_TIME:
                        cv2.imshow('Skeleton View', image)
                        cv2.imshow('Clean View', state.clean_image)
                        cv2.waitKey(1)
                        
                        logger.info(f"Guarding confirmed: {state.active_zone}")
                        
                        speak_audio("กรุณาบอกอาการของคุณได้เลยค่ะ")
                        patient_voice = listen_to_patient()
                        
                        try:
                            if patient_voice:
                                logger.info(f"Processing: zone={state.active_zone}, skin={state.current_skin_status}")
                                json_result = call_llama_reasoning(state.active_zone, state.current_skin_status, patient_voice)
                                logger.info(f"LLM response received: {len(json_result)} chars")
                                handle_output_guardrail(json_result)
                            else:
                                logger.warning("No voice input from patient")
                                speak_audio("ระบบไม่ได้ยินเสียงค่ะ ขออภัยในความไม่สะดวก")
                        
                        except Exception as e:
                            logger.error(f"Processing error: {e}", exc_info=True)
                            print(f"⚠️ [Processing Error]: {e}")
                            speak_audio("เกิดข้อผิดพลาด กรุณาติดต่อพยาบาลค่ะ")
                        
                        finally:
                            state.reset_zone()
                            time.sleep(1)
                
                if state.clean_image is not None:
                    cv2.imshow('Clean View', state.clean_image)
                cv2.imshow('Skeleton View', image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    logger.info("User pressed 'q' - exiting")
                    break
    
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        speak_audio("ระบบเกิดข้อผิดพลาด กรุณาติดต่อผู้ดูแล")
    
    finally:
        logger.info("Cleaning up resources")
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        logger.info("=== Smart Triage Kiosk Stopped ===")


if __name__ == "__main__":
    main()
