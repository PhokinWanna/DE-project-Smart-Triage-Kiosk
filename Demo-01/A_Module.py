import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import io
import json

pygame.mixer.init()

def listen_to_patient():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n" + "="*50)
        print("🎤 [ระบบกำลังฟัง] กรุณาพูดอาการของคุณ...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("🔄 [กำลังประมวลผลเสียง...]")
            text = recognizer.recognize_google(audio, language="th-TH")
            print(f"✅ ผู้ป่วยพูดว่า: '{text}'")
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            print("⚠️ [ไม่พบเสียงพูด หรือ ฟังไม่ชัดเจน]")
            return ""
        except Exception as e:
            print(f"⚠️ [STT Error]: {e}")
            return ""

def is_safe_for_public(text):
    sensitive_keywords = ["esi", "level", "โรค", "ประวัติ", "แพ้ยา", "ความดัน", "hiv", "เอดส์", "ฉุกเฉิน", "วิกฤต", "หัวใจ"]
    text_lower = text.lower()
    for keyword in sensitive_keywords:
        if keyword in text_lower:
            return False, keyword
    return True, None

def speak_audio(text):
    try:
        tts = gTTS(text=text, lang='th')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"⚠️ [TTS Error]: {e}")

def handle_output_guardrail(json_response_str):
    try:
        data = json.loads(json_response_str)
        clinical_summary = data.get("clinical_summary", "")
        esi_level = data.get("esi_level", "")
        
        print(f"🖥️ [DASHBOARD (Private)]: ESI = {esi_level} | Summary = {clinical_summary}")
        
        is_safe, trigger_word = is_safe_for_public(clinical_summary)
        if is_safe:
            speak_audio("ระบบบันทึกอาการเบื้องต้นของคุณเรียบร้อยแล้วค่ะ")
        else:
            print(f"⚠️ [PDPA ALERT]: ระงับการพูดข้อมูลละเอียดอ่อน (พบคำว่า '{trigger_word}')")
            speak_audio("รับทราบข้อมูลแล้วค่ะ กรุณานั่งรอพยาบาลเรียกชื่อเพื่อซักประวัติเพิ่มเติมนะคะ")
            
    except json.JSONDecodeError:
        print("⚠️ [JSON Error]: ไม่สามารถแยกข้อมูลจาก Llama ได้")
        speak_audio("เกิดข้อผิดพลาดในการประมวลผล กรุณาติดต่อพยาบาลค่ะ")