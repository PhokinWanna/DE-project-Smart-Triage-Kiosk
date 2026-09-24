"""
config.py - Central System Configuration & Kiosk FSM Parameters
Unified Smart Triage Kiosk System - Final Production Architecture
"""

import os
from enum import Enum

class KioskState(Enum):
    PREPARE_STANDBY = "PREPARE_STANDBY"       # Idle monitoring: Camera + Wake-word listener
    GREETING_LANG_SELECT = "GREETING"         # Bilingual introduction & language negotiation
    PDPA_CONSENT = "PDPA_CONSENT"             # Read PDPA policy and await verbal consent
    SYMPTOM_INTERVIEW = "INTERVIEW"           # Interactive OPQRST Probing & Active Nudging Loop
    FINALIZING_CASE = "FINALIZING"           # Holistic multi-modal synthesis & SOAP generation
    DISPATCH_AND_RESET = "DISPATCH"           # Enqueue to conveyor belt & reset kiosk
    MANUAL_ROUTING = "MANUAL_ROUTING"         # Direct routing to human nurse desk (e.g. PDPA denied)


class ExitReason(Enum):
    NORMAL_COMPLETE = "NORMAL_COMPLETE"               # Encounter concluded naturally
    EMERGENCY_INTERRUPT = "EMERGENCY_INTERRUPT"       # Acute distress detected mid-case (ESI 1-2)
    AMS_3_STRIKES = "AMS_3_STRIKES"                   # 3 consecutive off-topic strikes (ESI 2 AMS)
    PDPA_DENIED = "PDPA_DENIED"                       # Patient refused data collection
    ABANDONED_TIMEOUT = "ABANDONED_TIMEOUT"           # Patient walked away mid-encounter


class SystemConfig:
    # --- Video & Display Pipeline ---
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    SHOW_DUAL_WINDOWS = True                 # Clean Patient View + Diagnostic Skeleton View

    # --- Perception Thresholds (Scale-Invariant Torso Ratios) ---
    PRESENCE_FRAMES_TRIGGER = 15             # Frames to confirm patient presence (~0.5s)
    PRESENCE_DETECTION_CONFIRM_FRAMES = 15   # Alias for backward compatibility
    GESTURE_HOLD_SECONDS = 1.5               # Scratch Filter: Hold clutching pose > 1.5s
    SKIN_DUTY_CYCLE_SECONDS = 7.0            # Run CNN skin inference once every 7.0 seconds
    
    # Anatomical distance ratios relative to torso length (|Mid-Shoulder - Mid-Hip|)
    CHEST_PAIN_TORSO_RATIO = 0.35            # Wrist to Sternum relative to Torso length
    ABDOMEN_PAIN_TORSO_RATIO = 0.40          # Wrist to Navel relative to Torso length
    HEADACHE_HEAD_RATIO = 0.28               # Wrist to Temple/Nose relative to Torso length
    ELBOW_FLEXION_MAX_ANGLE = 115.0          # Degrees: Arm must be bent inward
    CROSSED_ARMS_WRIST_RATIO = 0.45          # Crossed Arms Filter ratio
    THINKING_FOREARM_MIN_ANGLE = 65.0        # Thinking Pose vertical forearm angle

    # --- TensorFlow Lite / CNN Skin Model ---
    TFLITE_SKIN_MODEL_PATH = "Train-dataset/Train-02-CNN/skin_classifier.tflite"
    SKIN_CONFIDENCE_THRESHOLD = 0.65         # Minimum probability to classify as RED_FLUSHING

    # --- Audio Subsystem: Wake-Word & Fast-Whisper ---
    WAKE_WORDS_TH = ["สวัสดี", "หวัดดี", "เริ่ม", "ช่วยด้วย", "ตรวจ"]
    WAKE_WORDS_EN = ["hello", "hi", "start", "help", "triage"]
    
    FASTER_WHISPER_MODEL = "base"            # Local Fast-Whisper model
    FASTER_WHISPER_COMPUTE = "int8"          # 8-bit quantization: lightweight (~600MB memory)
    SPEECH_PAUSE_THRESHOLD = 2.5             # Wait 2.5s of silence before finalizing utterance
    SPEECH_TIMEOUT = 10.0                    # Max wait time for patient speech initiation
    SPEECH_PHRASE_LIMIT = 20.0               # Max duration of a continuous speaking turn
    MIC_ENERGY_THRESHOLD = 300               # Audio sensitivity floor

    # --- Conversational Clinical Rules & OPQRST ---
    MAX_OFFTOPIC_STRIKES = 3                 # 3 strikes = Altered Mental Status (ESI Level 2)
    PROBE_MAX_TOKENS = 45                    # Strictly cap follow-up question generation latency
    
    # Stopping Intent Tokens
    STOP_PHRASES_TH = ["หมดแล้ว", "แค่นี้", "ไม่มีแล้ว", "พอแล้ว", "เสร็จแล้ว", "ไม่มีอะไรเพิ่ม", "มีเท่านี้"]
    STOP_PHRASES_EN = ["that's all", "thats all", "nothing else", "no more", "done", "finished", "i'm good"]

    # --- Audio Template Cache Directory ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_TEMPLATE_DIR = os.path.join(BASE_DIR, "assets", "audio")

    # Audio Template File Mapping
    TEMPLATES = {
        "th": {
            "greeting": "th_greeting.mp3",
            "pdpa_notice": "th_pdpa.mp3",
            "pdpa_denied": "th_pdpa_denied.mp3",
            "inquiry": "th_inquiry.mp3",
            "still_listening": "th_still_listening.mp3",
            "anything_else": "th_anything_else.mp3",
            "wrapup": "th_wrapup.mp3",
            "emergency_alert": "th_emergency_alert.mp3",
            "ams_alert": "th_ams_alert.mp3"
        },
        "en": {
            "greeting": "en_greeting.mp3",
            "pdpa_notice": "en_pdpa.mp3",
            "pdpa_denied": "en_pdpa_denied.mp3",
            "inquiry": "en_inquiry.mp3",
            "still_listening": "en_still_listening.mp3",
            "anything_else": "en_anything_else.mp3",
            "wrapup": "en_wrapup.mp3",
            "emergency_alert": "en_emergency_alert.mp3",
            "ams_alert": "en_ams_alert.mp3"
        }
    }

    # Text Sources for Audio Templates (Polite Clinical Tone)
    SCRIPT_TEXTS = {
        "th": {
            "greeting": "สวัสดีค่ะ ดิฉันคือระบบผู้ช่วยพยาบาลคัดกรองอัตโนมัติ กรุณาแจ้งอาการ หรือเลือกใช้งานเป็น ภาษาไทย หรือ ภาษาอังกฤษ ได้เลยค่ะ",
            "pdpa_notice": "เนื่องด้วยตาม พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล หรือ PDPA ระบบขออนุญาตบันทึกเสียงและภาพของการสนทนานี้เพื่อการคัดกรองทางการแพทย์ ท่านประสงค์ยินยอมหรือไม่คะ?",
            "pdpa_denied": "รับทราบค่ะ ระบบจะไม่บันทึกข้อมูลของท่านใดๆ กรุณาติดต่อเคาน์เตอร์พยาบาลด้านหน้าได้โดยตรง ขอบคุณค่ะ",
            "inquiry": "วันนี้มีอาการไม่สบายตรงไหน หรือมีอะไรให้ดิฉันช่วยดูแลคะ?",
            "still_listening": "ดิฉันยังฟังอยู่นะคะ เล่าต่อได้เลยค่ะ",
            "anything_else": "ระบบบันทึกข้อมูลเรียบร้อยค่ะ มีอาการผิดปกติอื่นเพิ่มเติมอีกไหมคะ?",
            "wrapup": "ระบบได้รวบรวมข้อมูลและอาการทั้งหมดและส่งไปยังพยาบาลเรียบร้อยแล้ว กรุณานั่งรอเรียกชื่อที่จุดพักคอยนะคะ",
            "emergency_alert": "ระบบตรวจพบอาการเข้าข่ายฉุกเฉินวิกฤต กำลังส่งสัญญาณเรียกพยาบาลเข้ามาดูแลทันที กรุณานั่งนิ่งๆ นะคะ",
            "ams_alert": "ระบบได้ส่งสัญญาณแจ้งเตือนพยาบาลให้เข้ามาดูแลท่านที่หน้าตู้คัดกรองแล้ว กรุณารอสักครู่นะคะ"
        },
        "en": {
            "greeting": "Hello, I am your automated triage nurse assistant system. Please select English or Thai to proceed.",
            "pdpa_notice": "Under privacy regulations, the system requests permission to record audio and video of this conversation for medical screening purposes. Do you consent?",
            "pdpa_denied": "Understood. No data will be stored. Please proceed directly to the nurse reception desk. Thank you.",
            "inquiry": "What symptoms or discomfort bring you to the emergency department today? What can I help you with?",
            "still_listening": "I am still listening. Please feel free to continue.",
            "anything_else": "I have noted that down. Do you have any additional symptoms or details to add?",
            "wrapup": "Your clinical case has been compiled and dispatched to the triage nurse. Please take a seat in the waiting area.",
            "emergency_alert": "Critical emergency signals detected. Alerting resuscitation nurses to your station immediately. Please remain seated.",
            "ams_alert": "Triage staff have been dispatched to assist you at the kiosk. Please wait a moment."
        }
    }

    # --- Reasoning Engine (Local Ollama Llama 3.2) ---
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = "llama3.2:latest"
    OLLAMA_TIMEOUT = 5.0                     # Hard timeout before deterministic fallback triggers

    # --- Background "Conveyor Belt" Outbox Spooler ---
    OUTBOX_SPOOL_DIR = os.path.join(BASE_DIR, "spool_outbox")
    NURSE_DASHBOARD_ENDPOINT = "http://127.0.0.1:8000/api/triage/submit_case"
    HTTP_TIMEOUT = 3.0
    CONVEYOR_RETRY_INTERVAL = 5.0            # Seconds between dispatch retry attempts