"""
Smart Triage Kiosk - Vision Module (Demo-03)
Author: Phokin Wanna (6630613024)

Geometric primitives, anatomical normalization, and edge LiteRT skin classification.
"""

import math
import numpy as np
import cv2

# Universal Edge LiteRT / TFLite Import
from ai_edge_litert.interpreter import Interpreter



# =====================================================================
# 1. YOUR ORIGINAL GEOMETRIC UTILITIES (PRESERVED INTACT)
# =====================================================================

def calculate_angle(a, b, c):
    """
    Calculate angle at vertex b formed by points a-b-c using dot product.
    More numerically stable than atan2 method.
    
    Args:
        a, b, c: Landmarks with .x, .y attributes (MediaPipe format)
        
    Returns:
        float: Angle in degrees (0-180)
    """
    a_vec = np.array([a.x, a.y])
    b_vec = np.array([b.x, b.y])
    c_vec = np.array([c.x, c.y])
    
    ba = a_vec - b_vec
    bc = c_vec - b_vec
    
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return float(angle)


def ccw(A, B, C):
    """
    Check if three points are in counter-clockwise order.
    Used for line segment intersection detection.
    """
    return (C - A) * (B[0] - A[0]) > (B - A) * (C[0] - A[0])


def check_intersection(A, B, C, D):
    """
    Check if line segments AB and CD intersect using CCW orientation test.
    Used to detect crossed arms (False Positive Filter 2).
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# =====================================================================
# 2. ANATOMICAL INVARIANCE (FAIL-SAFE CHEST CLUTCH CHECK)
# =====================================================================

def evaluate_chest_clutch(landmarks, mp_pose):
    """
    Evaluates Levine's sign / chest clutching with foreshortening fail-safe.
    
    Returns:
        is_clutching (bool): True if acute arm flexion is clutching the chest.
        norm_distance (float): Normalized wrist-to-chest ratio.
    """
    # Key landmark coordinates
    l_sh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    r_sh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
    l_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    r_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

    l_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
    r_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])

    # Dynamic anatomical center
    chest_center = (l_sh + r_sh) / 2.0
    hip_center = (l_hip + r_hip) / 2.0

    # Biacromial width & Torso length
    biacromial_width = np.linalg.norm(l_sh - r_sh)
    torso_length = np.linalg.norm(chest_center - hip_center)

    # Fail-safe denominator: prevents division blowup if patient is hunched
    l_ref = max(torso_length, 0.75 * biacromial_width, 1e-4)

    # Normalized Euclidean distance from wrist to chest
    d_left = np.linalg.norm(l_wrist - chest_center) / l_ref
    d_right = np.linalg.norm(r_wrist - chest_center) / l_ref
    min_dist = min(d_left, d_right)

    # Calculate elbow flexion using your numerically stable calculate_angle()
    l_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    r_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # Thresholds: Wrist within 42% of torso scale and elbow bent between 30° and 120°
    left_clutch = (d_left <= 0.42) and (30.0 <= l_angle <= 120.0)
    right_clutch = (d_right <= 0.42) and (30.0 <= r_angle <= 120.0)

    return (left_clutch or right_clutch), float(min_dist)


# =====================================================================
# 3. EDGE LITERUNTIME SKIN ANOMALY CLASSIFIER
# =====================================================================

class SkinClassifier:
    """
    Runs inference on facial ROI crops using LiteRT (skin_classifier.tflite).
    Zero protobuf dependency.
    """
    def __init__(self, model_path="Train-dataset/Train-02-CNN/skin_classifier.tflite"):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Dimensions expected by the trained model (e.g. 64x64)
        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape
        self.width = self.input_shape[2]

    def predict_roi(self, face_roi_bgr, threshold=0.70):
        """
        Takes cropped BGR face patch, normalizes, and returns prediction.
        
        Returns:
            is_abnormal (bool): True if probability >= threshold
            score (float): Confidence score (0.0 to 1.0)
        """
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return False, 0.0

        # Resize to model input dimensions
        resized = cv2.resize(face_roi_bgr, (self.width, self.height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Scale to [0.0, 1.0]
        input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

        # LiteRT Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

        is_abnormal = bool(prediction >= threshold)
        return is_abnormal, float(prediction)