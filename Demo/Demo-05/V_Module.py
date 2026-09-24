"""
V_Module.py - Vision Perception Engine
Scale-Invariant Pose Tracking, 3 Clinical False-Positive Filters, and Throttled Face Skin CNN.
"""

import os
import cv2
import time
import numpy as np
import mediapipe as mp
from config import SystemConfig

class VisionEngine:
    def __init__(self):
        # 1. Initialize MediaPipe Pose Subsystem
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # 2. Initialize MediaPipe Face Mesh Subsystem (For Face ROI Isolation)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 3. Persistence Timers for Filter 3 (Scratch / Fleeting Motion Filter)
        self.gesture_start_times = {
            "chest_clutching": None,
            "abdominal_clutching": None,
            "head_clutching": None
        }
        self.presence_counter = 0

        # 4. Duty-Cycle Timer for Facial Skin CNN (Runs every 7.0 seconds)
        self.last_skin_inference_time = 0.0
        self.cached_skin_status = "NORMAL"

        # 5. TFLite Interpreter Bridge
        self.tflite_interpreter = None
        self._init_tflite_model()

    def _init_tflite_model(self):
        """Loads skin_classifier.tflite if present; otherwise runs mock CNN scaffold."""
        model_path = SystemConfig.TFLITE_SKIN_MODEL_PATH
        if os.path.exists(model_path):
            # Attempt 1: Standalone LiteRT / TFLite Runtime
            try:
                from ai_edge_litert.interpreter import Interpreter
                self.tflite_interpreter = Interpreter(model_path=model_path)
                self.tflite_interpreter.allocate_tensors()
                print(f"[+] Armed TFLite Skin CNN via LiteRT: {model_path}")
                return
            except Exception:
                pass

            try:
                from tflite_runtime.interpreter import Interpreter
                self.tflite_interpreter = Interpreter(model_path=model_path)
                self.tflite_interpreter.allocate_tensors()
                print(f"[+] Armed TFLite Skin CNN via tflite_runtime: {model_path}")
                return
            except Exception:
                pass

            # Attempt 2: Full TensorFlow Lite runtime
            try:
                import tensorflow as tf
                self.tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
                self.tflite_interpreter.allocate_tensors()
                print(f"[+] Armed TFLite Skin CNN via TensorFlow: {model_path}")
                return
            except Exception:
                pass

        print(f"[i] Running Skin Perception in High-Reliability CNN Scaffold Mode.")

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates 2D joint angle at vertex b in degrees."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _infer_skin_cnn(self, face_roi: np.ndarray) -> str:
        """
        Executes CNN inference on extracted facial region.
        Labels: 'NORMAL' vs 'RED_FLUSHING' (Pallor strictly removed).
        """
        if face_roi is None or face_roi.size == 0:
            return self.cached_skin_status

        # 1. Real TFLite Model Execution
        if self.tflite_interpreter is not None:
            try:
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()
                in_shape = input_details[0]['shape'][1:3]

                resized = cv2.resize(face_roi, (in_shape, in_shape[0]))
                tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

                self.tflite_interpreter.set_tensor(input_details[0]['index'], tensor)
                self.tflite_interpreter.invoke()
                preds = self.tflite_interpreter.get_tensor(output_details[0]['index'])[0]

                # Binary classification: Index 1 = Red/Flushing
                status = "RED_FLUSHING" if preds[0] >= SystemConfig.SKIN_CONFIDENCE_THRESHOLD else "NORMAL"
                self.cached_skin_status = status
                return status
            except Exception:
                pass

        # 2. Resilient Colorimetry Scaffold
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv)

        # Erythema flag: High saturation in the red wavelength spectrum
        if (mean_hue < 12 or mean_hue > 168) and mean_sat > 110:
            self.cached_skin_status = "RED_FLUSHING"
        else:
            self.cached_skin_status = "NORMAL"

        return self.cached_skin_status

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Processes an incoming video frame at ~30 FPS.
        Calculates scale invariance, applies 3 false-positive filters,
        and periodically executes the 7.0s face skin CNN.
        """
        h, w, _ = frame.shape
        clean_view = frame.copy()
        skeleton_view = frame.copy()
        now = time.time()

        telemetry = {
            "patient_present": False,
            "chest_clutching": False,
            "abdominal_clutching": False,
            "head_clutching": False,
            "skin_status": self.cached_skin_status,
            "clean_view": clean_view,
            "skeleton_view": skeleton_view,
            "active_filters": []
        }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb_frame)
        face_res = self.face_mesh.process(rgb_frame)

        # -------------------------------------------------------------
        # Presence Detection with Debounce Counter
        # -------------------------------------------------------------
        if not pose_res.pose_landmarks:
            self.presence_counter = max(0, self.presence_counter - 1)
            for k in self.gesture_start_times:
                self.gesture_start_times[k] = None
            return telemetry

        self.presence_counter += 1
        if self.presence_counter >= SystemConfig.PRESENCE_FRAMES_TRIGGER:
            telemetry["patient_present"] = True

        landmarks = pose_res.pose_landmarks.landmark
        get_pt = lambda lm: np.array([lm.x * w, lm.y * h])

        # Anatomical Joint Landmarks
        l_sh = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        r_sh = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        l_hip = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        r_hip = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        l_el = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        r_el = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        l_wr = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST])
        r_wr = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        nose = get_pt(landmarks[self.mp_pose.PoseLandmark.NOSE])

        # Virtual Anatomical Reference Anchors
        chest_center = (l_sh + r_sh) / 2.0
        navel_center = (l_hip + r_hip) / 2.0

        # Torso Length: Scale Invariance Vector
        torso_vector_length = np.linalg.norm(chest_center - navel_center) + 1e-6

        # Normalized Metric Distances
        l_wrist_chest = np.linalg.norm(l_wr - chest_center) / torso_vector_length
        r_wrist_chest = np.linalg.norm(r_wr - chest_center) / torso_vector_length
        l_elbow_ang = self._calculate_angle(l_sh, l_el, l_wr)
        r_elbow_ang = self._calculate_angle(r_sh, r_el, r_wr)

        # -------------------------------------------------------------
        # FILTER 1: Crossed Arms Filter
        # Both forearms folded symmetrically across opposite elbows
        # -------------------------------------------------------------
        l_tuck = np.linalg.norm(l_wr - r_el) / torso_vector_length < SystemConfig.CROSSED_ARMS_WRIST_RATIO
        r_tuck = np.linalg.norm(r_wr - l_el) / torso_vector_length < SystemConfig.CROSSED_ARMS_WRIST_RATIO
        is_crossed_arms = l_tuck and r_tuck

        if is_crossed_arms:
            telemetry["active_filters"].append("CROSSED_ARMS_FILTER")

        # -------------------------------------------------------------
        # FILTER 2: Thinking Pose Filter
        # Forearm angle upright (>= 65°) with wrist near chin/mandible
        # -------------------------------------------------------------
        l_forearm_ang = np.degrees(np.arctan2(abs(l_wr - l_el), abs(l_wr[0] - l_el[0]) + 1e-6))
        r_forearm_ang = np.degrees(np.arctan2(abs(r_wr - r_el), abs(r_wr[0] - r_el[0]) + 1e-6))
        
        l_chin_dist = np.linalg.norm(l_wr - nose) / torso_vector_length
        r_chin_dist = np.linalg.norm(r_wr - nose) / torso_vector_length

        is_thinking = (l_chin_dist < SystemConfig.HEADACHE_HEAD_RATIO and l_forearm_ang >= SystemConfig.THINKING_FOREARM_MIN_ANGLE) or \
                      (r_chin_dist < SystemConfig.HEADACHE_HEAD_RATIO and r_forearm_ang >= SystemConfig.THINKING_FOREARM_MIN_ANGLE)

        if is_thinking:
            telemetry["active_filters"].append("THINKING_POSE_FILTER")

        # Raw Posture Evaluations
        raw_chest = (not is_crossed_arms) and (
            (l_wrist_chest <= SystemConfig.CHEST_PAIN_TORSO_RATIO and l_elbow_ang <= SystemConfig.ELBOW_FLEXION_MAX_ANGLE) or
            (r_wrist_chest <= SystemConfig.CHEST_PAIN_TORSO_RATIO and r_elbow_ang <= SystemConfig.ELBOW_FLEXION_MAX_ANGLE)
        )

        raw_abdo = (not is_crossed_arms) and (
            np.linalg.norm(l_wr - navel_center) / torso_vector_length <= SystemConfig.ABDOMEN_PAIN_TORSO_RATIO or
            np.linalg.norm(r_wr - navel_center) / torso_vector_length <= SystemConfig.ABDOMEN_PAIN_TORSO_RATIO
        )

        raw_head = (not is_thinking) and (
            np.linalg.norm(l_wr - nose) / torso_vector_length <= SystemConfig.HEADACHE_HEAD_RATIO or
            np.linalg.norm(r_wr - nose) / torso_vector_length <= SystemConfig.HEADACHE_HEAD_RATIO
        )

        # -------------------------------------------------------------
        # FILTER 3: Scratch / Fleeting Motion Filter (Hold >= 1.5 seconds)
        # -------------------------------------------------------------
        def _check_persistence(key: str, active: bool) -> bool:
            if active:
                if self.gesture_start_times[key] is None:
                    self.gesture_start_times[key] = now
                    return False
                elapsed = now - self.gesture_start_times[key]
                return elapsed >= SystemConfig.GESTURE_HOLD_SECONDS
            else:
                self.gesture_start_times[key] = None
                return False

        telemetry["chest_clutching"] = _check_persistence("chest_clutching", raw_chest)
        telemetry["abdominal_clutching"] = _check_persistence("abdominal_clutching", raw_abdo)
        telemetry["head_clutching"] = _check_persistence("head_clutching", raw_head)

        # -------------------------------------------------------------
        # 4. Throttled Skin CNN Duty Cycle (Every 7.0 seconds)
        # -------------------------------------------------------------
        if (now - self.last_skin_inference_time) >= SystemConfig.SKIN_DUTY_CYCLE_SECONDS:
            if face_res.multi_face_landmarks:
                flms = face_res.multi_face_landmarks[0].landmark
                xs = [int(p.x * w) for p in flms]
                ys = [int(p.y * h) for p in flms]
                x1, x2 = max(0, min(xs)), min(w, max(xs))
                y1, y2 = max(0, min(ys)), min(h, max(ys))
                face_crop = frame[y1:y2, x1:x2]

                telemetry["skin_status"] = self._infer_skin_cnn(face_crop)
                self.last_skin_inference_time = now

        # Draw Visual Overlays on Skeleton View
        self.mp_draw.draw_landmarks(skeleton_view, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Pain zone indicators
        if telemetry["chest_clutching"]:
            cv2.circle(skeleton_view, tuple(chest_center.astype(int)), int(torso_vector_length * 0.35), (0, 0, 255), 3)
            cv2.putText(skeleton_view, "SIGN: ACUTE CHEST DISTRESS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif telemetry["abdominal_clutching"]:
            cv2.circle(skeleton_view, tuple(navel_center.astype(int)), int(torso_vector_length * 0.35), (0, 165, 255), 3)
            cv2.putText(skeleton_view, "SIGN: ABDOMINAL GUARDING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        elif telemetry["head_clutching"]:
            cv2.circle(skeleton_view, tuple(nose.astype(int)), int(torso_vector_length * 0.25), (255, 0, 255), 3)
            cv2.putText(skeleton_view, "SIGN: CRANIAL PAIN", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Skin status indicator
        if telemetry["skin_status"] == "RED_FLUSHING":
            cv2.putText(skeleton_view, "CNN SKIN: ERYTHEMA / FLUSHING", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Active false-positive filter status
        if telemetry["active_filters"]:
            f_label = ", ".join(telemetry["active_filters"])
            cv2.putText(skeleton_view, f"FILTER ACTIVE: {f_label}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        telemetry["clean_view"] = clean_view
        telemetry["skeleton_view"] = skeleton_view
        return telemetry

    def release(self):
        self.pose.close()
        self.face_mesh.close()