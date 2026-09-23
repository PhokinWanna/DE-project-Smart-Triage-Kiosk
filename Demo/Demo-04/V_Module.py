"""
V_Module.py - Perception Engine
Pose Pain Gesture Detection (with False-Positive Filtering) + TFLite Face Skin Analysis.
"""

import os
import cv2
import time
import numpy as np
import mediapipe as mp
from config import SystemConfig

class VisionEngine:
    def __init__(self):
        # 1. MediaPipe Pose Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 2. Timing Trackers for Scratch / Fleeting Motion Filtering
        self.gesture_start_times = {
            "chest_clutching": None,
            "abdominal_clutching": None,
            "head_clutching": None
        }
        self.presence_counter = 0

        # 3. TFLite CNN Skin Model Loader (Scaffold)
        self.tflite_interpreter = None
        self._init_tflite_model()

    def _init_tflite_model(self):
        """Attempts to load skin_classifier.tflite if present on disk."""
        model_path = SystemConfig.TFLITE_SKIN_MODEL_PATH
        if os.path.exists(model_path):
            try:
                from ai_edge_litert.interpreter import Interpreter
                self.tflite_interpreter = Interpreter(msodel_path=model_path)
                self.tflite_interpreter.allocate_tensors()
                print(f"[+] Loaded TFLite Skin CNN Model from: {model_path}")
            except Exception as e:
                print(f"[!] Warning: TensorFlow Lite runtime failed to load ({e}). Using mock CNN scaffold.")
        else:
            print(f"[!] Info: TFLite model not found at {model_path}. Running CNN inference in scaffold mode.")

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates 2D joint angle at vertex b in degrees."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _run_skin_cnn(self, face_roi: np.ndarray) -> str:
        """
        Executes CNN inference on extracted facial region.
        Labels: 'NORMAL' vs 'RED_FLUSHING' (Pallor removed as requested).
        """
        if face_roi is None or face_roi.size == 0:
            return "NORMAL"

        # Real TFLite Inference Execution
        if self.tflite_interpreter is not None:
            try:
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()
                
                # Preprocess: Resize & normalize to
                in_shape = input_details[0]['shape'][1:3]
                resized = cv2.resize(face_roi, (in_shape, in_shape[0]))
                tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

                self.tflite_interpreter.set_tensor(input_details[0]['index'], tensor)
                self.tflite_interpreter.invoke()
                preds = self.tflite_interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Class 0: Normal, Class 1: Red/Flushing
                return "RED_FLUSHING" if preds >= SystemConfig.SKIN_CONFIDENCE_THRESHOLD else "NORMAL"
            except Exception:
                pass

        # Scaffold fallback: Check mean saturation/redness balance
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv)
        if (mean_hue < 12 or mean_hue > 168) and mean_sat > 110:
            return "RED_FLUSHING"
        return "NORMAL"

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Primary perception loop per video frame.
        Applies torso normalization and crossed arms, thinking, and scratch filters.
        """
        h, w, _ = frame.shape
        clean_view = frame.copy()
        skeleton_view = frame.copy()
        current_time = time.time()

        telemetry = {
            "patient_present": False,
            "chest_clutching": False,
            "abdominal_clutching": False,
            "head_clutching": False,
            "skin_status": "NORMAL",
            "clean_view": clean_view,
            "skeleton_view": skeleton_view,
            "active_filters": []
        }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb_frame)
        face_res = self.face_mesh.process(rgb_frame)

        if not pose_res.pose_landmarks:
            self.presence_counter = max(0, self.presence_counter - 1)
            # Reset duration counters
            for k in self.gesture_start_times:
                self.gesture_start_times[k] = None
            return telemetry

        # Confirm patient presence
        self.presence_counter += 1
        if self.presence_counter >= SystemConfig.PRESENCE_FRAMES_TRIGGER:
            telemetry["patient_present"] = True

        landmarks = pose_res.pose_landmarks.landmark
        get_pt = lambda lm: np.array([lm.x * w, lm.y * h])

        # Essential keypoints
        l_sh = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        r_sh = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        l_hip = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        r_hip = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        l_el = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        r_el = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        l_wr = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST])
        r_wr = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        nose = get_pt(landmarks[self.mp_pose.PoseLandmark.NOSE])

        # Virtual anatomic anchors
        chest_center = (l_sh + r_sh) / 2.0
        navel_center = (l_hip + r_hip) / 2.0
        torso_length = np.linalg.norm(chest_center - navel_center) + 1e-6

        # Normalized metric vectors
        l_wrist_chest = np.linalg.norm(l_wr - chest_center) / torso_length
        r_wrist_chest = np.linalg.norm(r_wr - chest_center) / torso_length
        l_elbow_ang = self._calculate_angle(l_sh, l_el, l_wr)
        r_elbow_ang = self._calculate_angle(r_sh, r_el, r_wr)

        # -----------------------------------------------------------------
        # FALSE POSITIVE FILTER 1: Crossed Arms Filter
        # -----------------------------------------------------------------
        # If both wrists are simultaneously folded across opposing sides near armpits
        l_cross = np.linalg.norm(l_wr - r_el) / torso_length < SystemConfig.CROSSED_ARMS_WRIST_RATIO
        r_cross = np.linalg.norm(r_wr - l_el) / torso_length < SystemConfig.CROSSED_ARMS_WRIST_RATIO
        is_crossed_arms = l_cross and r_cross

        if is_crossed_arms:
            telemetry["active_filters"].append("CROSSED_ARMS_IGNORED")

        # -----------------------------------------------------------------
        # FALSE POSITIVE FILTER 2: Thinking Pose Filter
        # -----------------------------------------------------------------
        # Wrist near mandible with nearly upright vertical forearm
        l_forearm_upright = abs(l_el[0] - l_wr[0]) < (abs(l_el - l_wr) * 0.7)
        r_forearm_upright = abs(r_el[0] - r_wr[0]) < (abs(r_el - r_wr) * 0.7)
        is_thinking = (np.linalg.norm(l_wr - nose) / torso_length < SystemConfig.HEADACHE_HEAD_RATIO and l_forearm_upright) or \
                      (np.linalg.norm(r_wr - nose) / torso_length < SystemConfig.HEADACHE_HEAD_RATIO and r_forearm_upright)

        if is_thinking:
            telemetry["active_filters"].append("THINKING_POSE_IGNORED")

        # Raw gesture trigger evaluations
        raw_chest = (not is_crossed_arms) and (
            (l_wrist_chest <= SystemConfig.CHEST_PAIN_TORSO_RATIO and l_elbow_ang <= SystemConfig.ELBOW_FLEXION_MAX_ANGLE) or
            (r_wrist_chest <= SystemConfig.CHEST_PAIN_TORSO_RATIO and r_elbow_ang <= SystemConfig.ELBOW_FLEXION_MAX_ANGLE)
        )
        raw_abdomen = (not is_crossed_arms) and (
            np.linalg.norm(l_wr - navel_center) / torso_length <= SystemConfig.ABDOMEN_PAIN_TORSO_RATIO or
            np.linalg.norm(r_wr - navel_center) / torso_length <= SystemConfig.ABDOMEN_PAIN_TORSO_RATIO
        )
        raw_head = (not is_thinking) and (
            np.linalg.norm(l_wr - nose) / torso_length <= SystemConfig.HEADACHE_HEAD_RATIO or
            np.linalg.norm(r_wr - nose) / torso_length <= SystemConfig.HEADACHE_HEAD_RATIO
        )

        # -----------------------------------------------------------------
        # FALSE POSITIVE FILTER 3: Scratch / Fleeting Motion Filter (Persistence)
        # -----------------------------------------------------------------
        def _evaluate_persistence(flag_name: str, raw_val: bool) -> bool:
            if raw_val:
                if self.gesture_start_times[flag_name] is None:
                    self.gesture_start_times[flag_name] = current_time
                    return False
                elapsed = current_time - self.gesture_start_times[flag_name]
                return elapsed >= SystemConfig.GESTURE_HOLD_SECONDS
            else:
                self.gesture_start_times[flag_name] = None
                return False

        telemetry["chest_clutching"] = _evaluate_persistence("chest_clutching", raw_chest)
        telemetry["abdominal_clutching"] = _evaluate_persistence("abdominal_clutching", raw_abdomen)
        telemetry["head_clutching"] = _evaluate_persistence("head_clutching", raw_head)

        # -----------------------------------------------------------------
        # Facial Crop & TFLite Skin CNN Inference
        # -----------------------------------------------------------------
        if face_res.multi_face_landmarks:
            flms = face_res.multi_face_landmarks[0].landmark
            xs = [int(p.x * w) for p in flms]
            ys = [int(p.y * h) for p in flms]
            x_min, x_max = max(0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0, min(ys)), min(h, max(ys))
            
            face_crop = frame[y_min:y_max, x_min:x_max]
            telemetry["skin_status"] = self._run_skin_cnn(face_crop)

        # Draw overlays on Skeleton View
        self.mp_draw.draw_landmarks(skeleton_view, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if telemetry["chest_clutching"]:
            cv2.putText(skeleton_view, "ALERT: CHEST PAIN (CONFIRMED)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif telemetry["abdominal_clutching"]:
            cv2.putText(skeleton_view, "ALERT: ABDOMINAL GUARDING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        elif telemetry["head_clutching"]:
            cv2.putText(skeleton_view, "ALERT: CRANIAL DISTRESS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        if telemetry["skin_status"] == "RED_FLUSHING":
            cv2.putText(skeleton_view, "SKIN: ERYTHEMA / FLUSHING DETECTED", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        telemetry["clean_view"] = clean_view
        telemetry["skeleton_view"] = skeleton_view
        return telemetry

    def release(self):
        self.pose.close()
        self.face_mesh.close()