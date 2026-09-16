"""
config.py - Central System Configuration & Kiosk FSM Parameters
Unified Smart Triage Kiosk System
"""

import os
from enum import Enum

class KioskState(Enum):
    IDLE = "IDLE"                         # Standby: Awaiting patient detection
    GREETING_LANG_SELECT = "GREETING"     # Bilingual introduction & language choice
    PDPA_CONSENT = "PDPA_CONSENT"         # Read PDPA policy and await consent
    SYMPTOM_INTERVIEW = "INTERVIEW"       # "What can I do for you?" + symptom capture
    FINALIZING_CASE = "FINALIZING"       # Scenario A/B/C + LLM ESI/SOAP evaluation
    DISPATCH_AND_RESET = "DISPATCH"       # Send bundle to nurse & reset
    MANUAL_ROUTING = "MANUAL_ROUTING"     # Route patient to human desk (e.g. PDPA denial)


class SystemConfig:
    # --- Video & Display Settings ---
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    SHOW_DUAL_WINDOWS = True             # True = Clean View + Skeleton Diagnostic View

    # --- Audio Engine (Lightweight Cloud API) ---
    SPEECH_PAUSE_THRESHOLD = 2.5         # Natural speech pause before finalizing utterance
    SPEECH_ENERGY_THRESHOLD = 300        # Baseline mic sensitivity
    SPEECH_TIMEOUT = 10.0                # Max wait time for speech initiation
    SPEECH_PHRASE_LIMIT = 20.0           # Max recording duration per turn

    # --- Reasoning Engine (Local Edge Ollama) ---
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = "llama3.2:latest"     # Llama 3.2 1B or 3B
    OLLAMA_TIMEOUT = 5.0                 # Seconds before deterministic fallback fires

    # --- Vision & False-Positive Filter Thresholds ---
    PRESENCE_FRAMES_TRIGGER = 15         # Consecutive frames to wake up kiosk (~0.5s)
    GESTURE_HOLD_SECONDS = 1.5           # Scratch filter: Must hold gesture > 1.5s to be real pain
    CHEST_PAIN_TORSO_RATIO = 0.35        # Wrist-to-sternum distance normalized by torso length
    ABDOMEN_PAIN_TORSO_RATIO = 0.40      # Wrist-to-navel distance normalized by torso length
    HEADACHE_HEAD_RATIO = 0.28           # Wrist-to-temple/nose distance normalized by torso length
    ELBOW_FLEXION_MAX_ANGLE = 115.0      # Degrees: Arm must be bent inward
    CROSSED_ARMS_WRIST_RATIO = 0.45      # Crossed arms filter: Wrists tucked near opposite armpits
    THINKING_FOREARM_MIN_ANGLE = 65.0    # Thinking pose filter: Forearm upright angle

    # --- CNN Skin Classification (TFLite Scaffold) ---
    TFLITE_SKIN_MODEL_PATH = "Train-dataset/Train-02-CNN/skin_classifier.tflite"
    SKIN_CONFIDENCE_THRESHOLD = 0.65     # Threshold to flag Erythema/Redness

    # --- Network & Nurse Dashboard ---
    NURSE_DASHBOARD_ENDPOINT = "http://127.0.0.1:8000/api/triage/submit_case"
    HTTP_TIMEOUT = 3.0

    # --- Bilingual Dialog Scripts ---
    SCRIPTS = {
        "th": {
            "greeting": "สวัสดีครับ ผมคือพยาบาลคัดกรองอัจฉริยะ กรุณาเลือกภาษา หรือพูดภาษาไทยได้เลยครับ. Hello, please choose your language.",
            "inquiry": "วันนี้มีอาการไม่สบายหรือมีอะไรให้ผมช่วยดูแลครับ?",
            "pdpa_notice": "ตาม พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล ทางโรงพยาบาลขออนุญาตบันทึกเสียงและภาพท่าทางเพื่อการประเมินความเร่งด่วนทางการแพทย์ ท่านยินยอมหรือไม่ครับ?",
            "pdpa_denied": "รับทราบครับ ระบบจะไม่บันทึกข้อมูลใดๆ กรุณาติดต่อเคาน์เตอร์พยาบาลด้านหน้าได้โดยตรงเลยครับ ขอบคุณครับ",
            "wrapup": "ระบบได้รวบรวมข้อมูลอาการและส่งไปยังโต๊ะพยาบาลเรียบร้อยแล้ว กรุณานั่งรอเรียกชื่อที่จุดพักคอยนะครับ",
            "voice_lang": "th"
        },
        "en": {
            "greeting": "Hello, I am your smart triage assistant. Please speak English or select your language.",
            "inquiry": "What brings you to the emergency department today? What can I do for you?",
            "pdpa_notice": "In compliance with personal data protection regulations, we need to collect your speech and posture signals for emergency assessment. Do you consent?",
            "pdpa_denied": "Understood. No data will be stored. Please proceed directly to the nurse reception desk for manual triage. Thank you.",
            "wrapup": "Your clinical case has been compiled and sent to the nurse station. Please take a seat in the waiting area.",
            "voice_lang": "en"
        }
    }