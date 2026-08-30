import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import io
import json

pygame.mixer.init()

def listen_to_patient():
    recognizer = sr.Recognizer()

    # --- FIX: ค่า default ของ pause_threshold คือ 0.8 วินาที
    # ถ้าผู้ป่วยหยุดพักระหว่างประโยคแค่ 0.8s ระบบจะตัดเสียงทันที
    # เพิ่มเป็น 2.5s เพื่อให้พูดได้ยาวขึ้นโดยไม่ถูกตัดกลางคัน
    recognizer.pause_threshold = 2.0         
    recognizer.non_speaking_duration = 2.5   # always = pause_threshold
    recognizer.phrase_threshold = 0.3        

    with sr.Microphone() as source:
        print("\n" + "="*50)
        print("🎤 [ระบบกำลังฟัง] กรุณาพูดอาการของคุณ...")
        # เพิ่ม duration เป็น 2s เพื่อ calibrate noise ในสภาพแวดล้อม รพ. ได้ดีขึ้น
        recognizer.adjust_for_ambient_noise(source, duration=2)
        try:
            # timeout=8  : รอผู้ป่วยเริ่มพูดได้นานขึ้น (เผื่อช้า/ตกใจ)
            # phrase_time_limit=60 : รองรับการพูดอาการยาวๆ ได้ถึง 60 วินาที
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=60)
            print("🔄 [กำลังประมวลผลเสียง...]")
            text = recognizer.recognize_google(audio, language="th-TH")
            print(f"✅ ผู้ป่วยพูดว่า: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("⚠️ [หมดเวลารอ] ผู้ป่วยไม่ได้พูดภายใน 8 วินาที")
            return ""
        except sr.UnknownValueError:
            print("⚠️ [ฟังไม่ชัดเจน] ไม่สามารถแปลงเสียงเป็นข้อความได้")
            return ""
        except Exception as e:
            print(f"⚠️ [STT Error]: {e}")
            return ""

def is_safe_for_public(text):
    sensitive_keywords = [
        "esi", "level", "โรค", "ประวัติ", "แพ้ยา", "ความดัน",
        "hiv", "เอดส์", "วิกฤต",
    ]

    # คำที่ block เฉพาะเมื่อไม่มี clinical context ขยายความ
    clinical_whitelist_patterns = [
        ("หัวใจ",   ["cardiac", "concern", "chest", "หน้าอก", "guarding", "observation", "triage", "หัวใจเต้น"]),
        ("ฉุกเฉิน", ["triage", "assessment", "observation", "clinical", "guarding"]),
    ]

    text_lower = text.lower()

    for keyword in sensitive_keywords:
        if keyword in text_lower:
            return False, keyword

    for sensitive_word, allowed_context_words in clinical_whitelist_patterns:
        if sensitive_word in text_lower:
            is_clinical_context = any(ctx in text_lower for ctx in allowed_context_words)
            if not is_clinical_context:
                return False, sensitive_word

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