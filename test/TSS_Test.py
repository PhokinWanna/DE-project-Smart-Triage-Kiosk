import time

class TriageAudioController:
    def __init__(self):
        # คำต้องห้าม (Sensitive Keywords) ถ้าเจอคำพวกนี้ ห้ามพูดออกลำโพงเด็ดขาด!
        self.sensitive_keywords = [
            "esi", "level", "โรค", "ประวัติ", "แพ้ยา", "ความดัน", 
            "hiv", "เอดส์", "ฉุกเฉิน", "วิกฤต", "หัวใจ"
        ]
        print("[System] Audio Controller Initialized with PDPA Guardrail.")

    def is_safe_for_public(self, text):
        """ด่านตรวจ PDPA: ตรวจสอบว่ามีคำละเอียดอ่อนหรือไม่"""
        text_lower = text.lower()
        for keyword in self.sensitive_keywords:
            if keyword in text_lower:
                return False, keyword # เจอคำต้องห้าม
        return True, None # ปลอดภัย

    def speak(self, text):
        """ฟังก์ชันสำหรับออกเสียง (TTS Engine)"""
        # --- สำหรับเทอมหน้า: ตรงนี้เราจะใส่โค้ด VITS TTS แบบ Offline ---
        # แต่ตอนนี้เราจำลองการทำงานไปก่อน
        print(f"\n🔊 [KIOSK SPEAKER BROADCASTING]: '{text}'")
        time.sleep(2) # จำลองเวลาพูด

    def process_llm_output(self, llm_response, esi_level):
        """รับข้อมูลจาก LLM และตัดสินใจว่าจะทำอย่างไร"""
        print("\n" + "="*40)
        print("📥 [RECEIVED DATA FROM LLM]")
        print(f"ESI Level: {esi_level}")
        print(f"Summary: {llm_response}")
        print("="*40)

        # 1. ส่งข้อมูลทั้งหมดเข้า Nurse Dashboard เสมอ (Private)
        print("🖥️ [SENDING TO NURSE DASHBOARD]: ข้อมูลครบถ้วน ถูกส่งเข้าระบบพยาบาลแล้ว")

        # 2. ตรวจสอบก่อนพูดออกลำโพง (PDPA Guardrail)
        is_safe, trigger_word = self.is_safe_for_public(llm_response)

        if is_safe:
            # ถ้าปลอดภัย ให้ Kiosk พูดสรุปสั้นๆ ได้
            self.speak(llm_response)
        else:
            # ถ้าไม่ปลอดภัย ระบบ Kiosk จะพูดประโยค Default (Safe Prompt) แทน
            print(f"⚠️ [PDPA ALERT]: พบคำละเอียดอ่อน '{trigger_word}' -> ระงับการออกเสียงข้อมูลจริง")
            self.speak("ระบบบันทึกอาการของคุณเรียบร้อยแล้วค่ะ กรุณานั่งรอพยาบาลเรียกชื่อสักครู่นะคะ")

# --- ลองทดสอบระบบ ---
if __name__ == "__main__":
    audio_sys = TriageAudioController()

    # เคสที่ 1: ข้อความทั่วไป (LLM สรุปอาการทั่วๆ ไป)
    print("\n--- TEST CASE 1: Normal Symptom ---")
    llm_output_1 = "ผู้ป่วยมีอาการปวดหัวตึบๆ นอนพักไม่เพียงพอ"
    audio_sys.process_llm_output(llm_output_1, esi_level=4)

    # เคสที่ 2: ข้อความอันตราย/ข้อมูลส่วนตัว (LLM เผลอหลุดวิเคราะห์โรค)
    print("\n--- TEST CASE 2: Sensitive Data ---")
    llm_output_2 = "ผู้ป่วยมีอาการแน่นหน้าอกรุนแรง สงสัยว่าจะเป็นโรคหัวใจกำเริบฉับพลัน (ESI 2)"
    audio_sys.process_llm_output(llm_output_2, esi_level=2)