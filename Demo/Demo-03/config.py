"""
Configuration file for Smart Triage Kiosk System
Centralized settings for Vision, Audio, Reasoning, and Logging
"""
import os
import logging

# ============================================================================
# VISION MODULE CONFIGURATION
# ============================================================================
CONFIRMATION_TIME = 2.0
CALIBRATION_TIME = 3.0

SKIN_PALE_THRESHOLD = -4.5
SKIN_RED_THRESHOLD = 4.5
SKIN_ANALYSIS_ROI_SIZE = 15

CHEST_RATIO_TO_BODY = 0.55
HEAD_WIDTH_RATIO = 0.3
HEAD_HEIGHT_RATIO = 0.58
MOUTH_TOUCH_THRESHOLD_RATIO = 0.30
ELBOW_SUPPORT_THRESHOLD_RATIO = 0.25

ARM_ANGLE_MIN = 60
ARM_ANGLE_MAX = 115
HEAD_SCRATCH_Z_THRESHOLD = 0.02

# ============================================================================
# AUDIO MODULE CONFIGURATION
# ============================================================================
STT_TIMEOUT = 8
STT_PHRASE_LIMIT = 60
STT_PAUSE_THRESHOLD = 2.0
STT_NON_SPEAKING_DURATION = 2.5
STT_PHRASE_THRESHOLD = 0.3
STT_LANGUAGE = "th-TH"

TTS_LANGUAGE = "th"
TTS_TIMEOUT = 10

# ============================================================================
# REASONING MODULE CONFIGURATION
# ============================================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_TIMEOUT = 30
OLLAMA_CONNECTION_TIMEOUT = 5

ESI_ZONE_MINIMUMS = {"Head": 3, "Chest": 2, "Abdomen": 3}
DEFAULT_ESI_LEVEL = 3

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = logging.INFO
LOG_FILE = "triage_kiosk.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# PRIVACY & SECURITY CONFIGURATION
# ============================================================================
SENSITIVE_KEYWORDS = [
    "esi", "level", "โรค", "ประวัติ", "แพ้ยา", "ความดัน",
    "hiv", "เอดส์", "วิกฤต",
]

CLINICAL_WHITELIST_PATTERNS = [
    ("หัวใจ", ["cardiac", "concern", "chest", "หน้าอก", "guarding", "observation", "triage", "หัวใจเต้น"]),
    ("ฉุกเฉิน", ["triage", "assessment", "observation", "clinical", "guarding"]),
]

INJECTION_PATTERNS = ["```", "[SYSTEM:", "IGNORE", "override"]

# ============================================================================
# CAMERA & UI CONFIGURATION
# ============================================================================
CAMERA_ID = 0
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5

DEBUG_PANEL_WIDTH = 290
DEBUG_PANEL_HEIGHT = 120
DEBUG_PANEL_OFFSET_X = 15
DEBUG_PANEL_OFFSET_Y = 15
DEBUG_FONT_SCALE = 0.48

# ============================================================================
# FEATURE FLAGS
# ============================================================================
ENABLE_LOGGING = True
ENABLE_DEBUG_DISPLAY = True
SHOW_SKELETON_LANDMARKS = True
SHOW_ZONE_POLYGONS = True
