"""
Audio Module - STT, TTS, and privacy-aware output
Critical Improvements:
- Robust JSON parsing with validation
- Better error handling and fallbacks
- Non-blocking TTS option
- Improved PDPA privacy filtering
"""
import speech_recognition as sr
from gtts import gTTS
import pygame
import io
import json
import logging
import re
import time

import config

logger = logging.getLogger(__name__)
pygame.mixer.init()


def listen_to_patient():
    """Listen to patient symptoms using Speech-to-Text (Google API)."""
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = config.STT_PAUSE_THRESHOLD
    recognizer.non_speaking_duration = config.STT_NON_SPEAKING_DURATION
    recognizer.phrase_threshold = config.STT_PHRASE_THRESHOLD
    
    with sr.Microphone() as source:
        print("\n" + "=" * 50)
        print("🎤 [ระบบกำลังฟัง] กรุณาพูดอาการของคุณ...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio = recognizer.listen(
                source,
                timeout=config.STT_TIMEOUT,
                phrase_time_limit=config.STT_PHRASE_LIMIT
            )
            print("🔄 [กำลังประมวลผลเสียง...]")
            text = recognizer.recognize_google(audio, language=config.STT_LANGUAGE)
            print(f"✅ ผู้ป่วยพูดว่า: '{text}'")
            logger.info(f"STT recognized: {text[:100]}")
            return text
        
        except sr.WaitTimeoutError:
            logger.warning("STT timeout - patient did not speak within time limit")
            print("⚠️ [หมดเวลารอ] ผู้ป่วยไม่ได้พูดภายใน 8 วินาที")
            return ""
        except sr.UnknownValueError:
            logger.warning("STT could not understand audio")
            print("⚠️ [ฟังไม่ชัดเจน] ไม่สามารถแปลงเสียงเป็นข้อความได้")
            return ""
        except Exception as e:
            logger.error(f"STT error: {e}")
            print(f"⚠️ [STT Error]: {e}")
            return ""


def speak_audio(text, timeout=None):
    """Play audio with optional timeout (blocking mode)."""
    timeout = timeout or config.TTS_TIMEOUT
    try:
        tts = gTTS(text=text, lang=config.TTS_LANGUAGE)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp, 'mp3')
        pygame.mixer.music.play()
        logger.debug(f"TTS playing: {text[:50]}")
        
        start_time = time.time()
        while pygame.mixer.music.get_busy():
            if time.time() - start_time > timeout:
                pygame.mixer.music.stop()
                logger.warning("TTS playback timeout")
                break
            pygame.time.Clock().tick(10)
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        print(f"⚠️ [TTS Error]: {e}")


def is_safe_for_public(text):
    """
    Check if clinical summary is safe to announce publicly (PDPA compliance).
    Returns: tuple (is_safe: bool, trigger_word: str or None)
    """
    text_lower = text.lower()
    
    for keyword in config.SENSITIVE_KEYWORDS:
        if keyword in text_lower:
            logger.warning(f"PDPA alert: sensitive keyword detected: {keyword}")
            return False, keyword
    
    for sensitive_word, allowed_context_words in config.CLINICAL_WHITELIST_PATTERNS:
        if sensitive_word in text_lower:
            is_clinical_context = any(ctx in text_lower for ctx in allowed_context_words)
            if not is_clinical_context:
                logger.warning(f"PDPA alert: {sensitive_word} without clinical context")
                return False, sensitive_word
    
    logger.info("Output cleared for public announcement")
    return True, None


def parse_llm_response(json_response_str):
    """Parse and validate LLM response JSON."""
    try:
        data = json.loads(json_response_str)
        
        esi_level = data.get("esi_level", "")
        clinical_summary = data.get("clinical_summary", "")
        
        if not isinstance(esi_level, int) or esi_level < 1 or esi_level > 5:
            logger.warning(f"Invalid ESI level: {esi_level}")
            esi_level = config.DEFAULT_ESI_LEVEL
        
        if not isinstance(clinical_summary, str):
            clinical_summary = str(clinical_summary)
        if len(clinical_summary.strip()) == 0:
            clinical_summary = "Patient assessment pending"
        
        return True, esi_level, clinical_summary
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return False, config.DEFAULT_ESI_LEVEL, "Error processing assessment"


def handle_output_guardrail(json_response_str):
    """Process LLM response with privacy guardrails (PDPA compliance)."""
    success, esi_level, clinical_summary = parse_llm_response(json_response_str)
    
    if not success:
        logger.error("Failed to parse LLM response")
        speak_audio("เกิดข้อผิดพลาดในการประมวลผล กรุณาติดต่อพยาบาลค่ะ")
        return
    
    print(f"🖥️ [DASHBOARD (Private)]: ESI = {esi_level} | Summary = {clinical_summary}")
    logger.info(f"ESI Level: {esi_level} | Summary length: {len(clinical_summary)}")
    
    is_safe, trigger_word = is_safe_for_public(clinical_summary)
    
    if is_safe:
        logger.info("Output approved for public announcement")
        speak_audio("ระบบบันทึกอาการเบื้องต้นของคุณเรียบร้อยแล้วค่ะ")
    else:
        logger.warning(f"Output blocked due to sensitive keyword: {trigger_word}")
        print(f"⚠️ [PDPA ALERT]: ระงับการพูดข้อมูลละเอียดอ่อน (พบคำว่า '{trigger_word}')")
        speak_audio("รับทราบข้อมูลแล้วค่ะ กรุณานั่งรอพยาบาลเรียกชื่อเพื่อซักประวัติเพิ่มเติมนะคะ")
