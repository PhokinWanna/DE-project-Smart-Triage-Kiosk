"""
R_Module.py - Clinical Reasoning Engine (Tuned Bedside Manner)
Echo-and-Pivot Probing, Context-Aware Relevance, Scenario A/B/C Matrix, and SOAP Synthesis.
"""

import json
import time
import requests
from config import SystemConfig, ExitReason

class ReasoningEngine:
    def __init__(self):
        self.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"
        self.warmup()

    def warmup(self):
        """Pre-warms Llama 3.2 in VRAM on boot so patient #1 experiences 0s delay."""
        print("[*] Pre-warming Llama 3.2 in VRAM (Pre-Flight Test)...")
        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": "ok",
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": -1 # Instructs Ollama to keep weights pinned in VRAM permanently
        }
        try:
            requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TIMEOUT)
            print("[+] Llama 3.2 is hot and pinned in VRAM.")
        except Exception as e:
            print(f"[!] Warmup Notice: Ollama not reachable on boot: {e}")

    # -------------------------------------------------------------------------
    # 1. STOPPING INTENT DETECTOR
    # -------------------------------------------------------------------------
    @staticmethod
    def is_stopping_phrase(utterance: str, lang: str = "th") -> bool:
        """Detects if patient indicates they are done explaining."""
        text = utterance.strip().lower()
        if len(text) == 0:
            return False

        phrases = SystemConfig.STOP_PHRASES_TH if lang == "th" else SystemConfig.STOP_PHRASES_EN
        return any(p in text for p in phrases)

    # -------------------------------------------------------------------------
    # 2. MID-SESSION ACUTE PREEMPTION CHECK (EMERGENCY BRAKE)
    # -------------------------------------------------------------------------
    @staticmethod
    def check_acute_preemption(utterance: str, visual_signs: dict) -> tuple[bool, int]:
        """Scans for sudden life-threatening collapse mid-case."""
        text = utterance.lower()
        chest_v = visual_signs.get("chest_clutching", False)

        critical_words = [
            "จะตาย", "หายใจไม่ออก", "หมดสติ", "วูบ", "แน่นหน้าอกมาก", "หัวใจจะหยุด",
            "cannot breathe", "passing out", "crushing chest", "heart attack", "unconscious"
        ]
        has_critical_vocal = any(w in text for w in critical_words)

        if "หมดสติ" in text or "unconscious" in text or "cannot breathe" in text:
            return True, 1

        if chest_v or has_critical_vocal:
            return True, 2

        return False, 0

    # -------------------------------------------------------------------------
    # 3. CONVERSATIONAL PROBING WITH ECHO-AND-PIVOT ("ทวนคำพูด + ตีกลับ")
    # -------------------------------------------------------------------------
    def evaluate_turn_and_probe(self, utterance: str, history: list, current_strikes: int, lang: str = "th") -> dict:
        """
        Evaluates turn relevance with context-awareness (recognizes time/numbers).
        Implements Echo-and-Pivot for off-topic turns and OPQRST follow-ups.
        """
        text = utterance.strip().lower()

        # Clinical symptom keywords
        medical_keywords = [
            "ปวด", "เจ็บ", "แน่น", "ไข้", "หนาว", "เวียน", "อ้วก", "ท้อง", "หัว", "แผล", "หายใจ", "ยา", "หมอ",
            "pain", "ache", "fever", "chest", "head", "stomach", "dizzy", "breath", "sick", "wound", "doctor"
        ]

        # Context-Aware Check: Did patient mention symptoms OR answer with time/quantity?
        has_medical_term = any(kw in text for kw in medical_keywords)
        has_time_or_number = any(kw in text for kw in SystemConfig.TIME_AND_NUMERIC_KEYWORDS)

        is_relevant = has_medical_term or has_time_or_number or len(text.split()) >= 4

        # ---------------------------------------------------------------------
        # Case A: Off-Topic / Disoriented Utterance
        # ---------------------------------------------------------------------
        if not is_relevant and len(text) > 0:
            new_strikes = current_strikes + 1
            print(f"[!] Off-Topic Detected. Strike {new_strikes}/{SystemConfig.MAX_OFFTOPIC_STRIKES}")

            # Strike 3: Altered Mental Status
            if new_strikes >= SystemConfig.MAX_OFFTOPIC_STRIKES:
                return {
                    "is_off_topic": True,
                    "strikes": new_strikes,
                    "ams_triggered": True,
                    "action": "TRIGGER_AMS",
                    "speech_output": SystemConfig.SCRIPT_TEXTS[lang]["ams_alert"],
                    "use_template": True,
                    "template_key": "ams_alert"
                }

            # Strikes 1 & 2: Dynamic Echo-and-Pivot Prompt (ทวนคำพูด + ตีกลับ)
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือพยาบาลคัดกรองหญิงประจำห้องฉุกเฉิน
หน้าที่ของคุณ:
1. ตอบรับสิ่งที่คนไข้เพิ่งพูดสั้นๆ (ทวนคำพูดไม่เกิน 5 คำ เช่น "รับทราบค่ะว่า...")
2. ดึงบทสนทนากลับมาถามอาการเจ็บป่วยหลักของคนไข้ทันที
3. ใช้สรรพนามแทนตัวเองว่า "ดิฉัน" และลงท้ายด้วย "ค่ะ" หรือ "นะคะ" เท่านั้น
4. ห้ามใช้คำว่า "ครับ" เด็ดขาด
5. ความยาวทั้งหมดต้องไม่เกิน 1 ประโยคสั้นๆ (ห้ามเกิน 20 คำ)
<|eot_id|><|start_header_id|>user<|end_header_id|>
คนไข้พูดนอกเรื่องว่า: "{utterance}"
จงสร้างประโยคตอบรับและดึงกลับมาเรื่องอาการ:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            payload = {
                "model": SystemConfig.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": SystemConfig.PROBE_MAX_TOKENS
                },
                "keep_alive": -1
            }

            try:
                res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TIMEOUT)
                if res.status_code == 200:
                    pivot_text = res.json().get("response", "").strip()
                    # Sanitize any accidental masculine particles
                    pivot_text = pivot_text.replace("ครับ", "ค่ะ").replace('"', '')
                    return {
                        "is_off_topic": True,
                        "strikes": new_strikes,
                        "ams_triggered": False,
                        "action": "STEER_PIVOT",
                        "speech_output": pivot_text,
                        "use_template": False
                    }
            except Exception:
                pass

            # Safe static fallback pivot
            safe_fallback = f"รับทราบค่ะว่า{utterance} แต่ตอนนี้รบกวนช่วยบอกอาการเจ็บป่วยหลักๆ ให้ดิฉันทราบก่อนนะคะ" if lang == "th" else \
                            f"I understand, but please focus on telling me your main medical symptoms first."
            return {
                "is_off_topic": True,
                "strikes": new_strikes,
                "ams_triggered": False,
                "action": "STEER_PIVOT",
                "speech_output": safe_fallback,
                "use_template": False
            }

        # ---------------------------------------------------------------------
        # Case B: On-Topic Utterance -> OPQRST Completeness Check
        # ---------------------------------------------------------------------
        conversation_context = " ".join([turn.get("text", "") for turn in history if turn.get("role") == "patient"])
        combined_text = f"{conversation_context} {utterance}".strip()

        has_timing = any(w in combined_text for w in ["เมื่อวาน", "กี่โมง", "โมง", "วัน", "ชั่วโมง", "นาที", "yesterday", "hours", "days", "since"])
        has_severity = any(w in combined_text for w in ["มาก", "น้อย", "เต็ม 10", "10", "severe", "mild", "moderate", "scale", "ทนได้"])

        # If timing and severity are both answered, patient has provided full clinical picture
        if has_timing and has_severity:
            return {
                "is_off_topic": False,
                "strikes": current_strikes,
                "ams_triggered": False,
                "action": "DETAILS_SUFFICIENT",
                "speech_output": "",
                "use_template": True
            }

        # Dynamic OPQRST Probing via Llama 3.2
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือพยาบาลคัดกรองหญิงประจำห้องฉุกเฉิน
หน้าที่ของคุณ: ถามคำถามคัดกรองตามหลัก OPQRST เพิ่มเติม 1 คำถามสั้นๆ
กฎเหล็ก:
1. ถามเฉพาะจุดที่ยังขาด: เวลาเริ่มต้นมีอาการ (Onset) หรือ ระดับความปวดเต็ม 10 ให้เท่าไหร่ (Severity)
2. ห้ามทายหรือพูดชื่อโรคเด็ดขาด
3. ห้ามเอ่ยชื่อหรือข้อมูลส่วนตัว
4. ใช้สรรพนาม "ดิฉัน" และลงท้ายด้วย "ค่ะ" หรือ "นะคะ" เท่านั้น ห้ามใช้ "ครับ"
5. ความยาวไม่เกิน 1 ประโยคสั้นๆ (ไม่เกิน 15 คำ)
<|eot_id|><|start_header_id|>user<|end_header_id|>
คนไข้แจ้งอาการ: "{combined_text}"
จงสร้างคำถามคัดกรองสั้นๆ 1 คำถาม:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": SystemConfig.PROBE_MAX_TOKENS
            },
            "keep_alive": -1
        }

        try:
            res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TIMEOUT)
            if res.status_code == 200:
                probe_q = res.json().get("response", "").strip()
                probe_q = probe_q.replace("ครับ", "ค่ะ").replace('"', '').replace('\n', ' ')
                return {
                    "is_off_topic": False,
                    "strikes": current_strikes,
                    "ams_triggered": False,
                    "action": "PROBE_QUESTION",
                    "speech_output": probe_q,
                    "use_template": False
                }
        except Exception:
            pass

        # Deterministic fallback probe
        fallback_probe = "เป็นมานานกี่ชั่วโมงแล้วคะ และเจ็บมากไหม เต็ม 10 ให้เท่าไหร่คะ?" if lang == "th" else \
                         "How long have you had this, and how severe is it on a scale of 1 to 10?"
        return {
            "is_off_topic": False,
            "strikes": current_strikes,
            "ams_triggered": False,
            "action": "PROBE_QUESTION",
            "speech_output": fallback_probe,
            "use_template": False
        }

    # -------------------------------------------------------------------------
    # 4. SCENARIO A / B / C CROSS-VALIDATION MATRIX
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_scenario(verbal_transcript: str, visual_signs: dict) -> dict:
        """Cross-validates kinetic postures against stated complaints."""
        text = verbal_transcript.lower()
        chest_v = visual_signs.get("chest_clutching", False)
        abdo_v = visual_signs.get("abdominal_clutching", False)
        head_v = visual_signs.get("head_clutching", False)

        has_chest_vocal = any(k in text for k in ["chest", "heart", "เจ็บหน้าอก", "แน่นหน้าอก", "หัวใจ"])
        has_abdo_vocal = any(k in text for k in ["stomach", "abdomen", "belly", "ปวดท้อง", "เสียดท้อง", "จุกแน่น"])
        has_head_vocal = any(k in text for k in ["headache", "dizzy", "stroke", "ปวดหัว", "เวียนหัว", "มึนหัว", "ศีรษะ"])
        has_denial = any(k in text for k in ["fine", "nothing", "okay", "no pain", "ไม่เป็นไร", "สบายดี", "นิดหน่อย"])

        # Scenario A: Mutual Confirmation
        if (chest_v and has_chest_vocal) or (abdo_v and has_abdo_vocal) or (head_v and has_head_vocal):
            return {
                "scenario_type": "SCENARIO_A_CONFIRMED",
                "clinical_flag": "HIGH_CONFIDENCE_ALIGNMENT",
                "notes": "Patient physical pain posturing directly corroborates stated complaint."
            }

        # Scenario C: Visual distress with verbal denial/silence (Stoic patient)
        if (chest_v or abdo_v) and (has_denial or len(text) == 0):
            return {
                "scenario_type": "SCENARIO_C_CLINICAL_CONFLICT",
                "clinical_flag": "SAFETY_OVERRIDE_STOIC_PATIENT",
                "notes": "Patient clutching vital pain zone while verbally minimizing symptoms. Severe distress suspected."
            }

        # Scenario B: Subjective complaint only
        return {
            "scenario_type": "SCENARIO_B_SUBJECTIVE_ONLY",
            "clinical_flag": "STANDARD_VERBAL_COMPLAINT",
            "notes": "Symptoms reported verbally without active kinetic clutching."
        }

    # -------------------------------------------------------------------------
    # 5. DETERMINISTIC EMERGENCY SAFETY FALLBACK
    # -------------------------------------------------------------------------
    @staticmethod
    def _deterministic_fallback(transcript: str, visual_signs: dict, scenario: dict, skin_status: str, exit_reason: ExitReason) -> dict:
        """Hardcoded ESI v4 rules if Ollama is unreachable."""
        sc_type = scenario.get("scenario_type")
        exit_val = exit_reason.value if hasattr(exit_reason, "value") else str(exit_reason)

        if exit_val == ExitReason.AMS_3_STRIKES.value:
            return {
                "esi_level": 2,
                "urgency": "Emergent / High Risk",
                "scenario_type": sc_type,
                "subjective": f"Patient exhibiting persistent conversational confusion: \"{transcript}\"",
                "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}, Strike Count: 3",
                "assessment": "Suspected acute Altered Mental Status (AMS). High risk for neurological or metabolic deficit.",
                "plan": "Immediate nursing evaluation at kiosk, blood glucose check, acute stroke screen.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

        if exit_val == ExitReason.EMERGENCY_INTERRUPT.value or sc_type in ["SCENARIO_A_CONFIRMED", "SCENARIO_C_CLINICAL_CONFLICT"] or visual_signs.get("chest_clutching", False):
            return {
                "esi_level": 2,
                "urgency": "Emergent / High Risk",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "Suspected acute coronary syndrome or critical thoracic distress.",
                "plan": "Immediate 12-lead ECG, priority transfer to monitored resuscitation bed.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

        if visual_signs.get("abdominal_clutching", False):
            return {
                "esi_level": 3,
                "urgency": "Urgent",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "Acute abdominal pain with physical guarding, multiple diagnostic resources anticipated.",
                "plan": "Transfer to acute care bay, establish IV access, order routine abdominal panel.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

        return {
            "esi_level": 4,
            "urgency": "Less Urgent",
            "scenario_type": sc_type,
            "subjective": transcript,
            "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}",
            "assessment": "Stable ambulatory presentation. Single diagnostic resource anticipated.",
            "plan": "Standard outpatient triage queue.",
            "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
        }

    # -------------------------------------------------------------------------
    # 6. FINAL CLINICAL SYNTHESIS
    # -------------------------------------------------------------------------
    def evaluate_final_case(self, conversation_history: list, visual_signs: dict, skin_status: str, exit_reason: ExitReason, language: str) -> dict:
        """Synthesizes the complete patient encounter into the final ESI Level + SOAP note."""
        transcript = " ".join([t["text"] for t in conversation_history if t.get("role") == "patient"])
        scenario = self.evaluate_scenario(transcript, visual_signs)

        exit_val = exit_reason.value if hasattr(exit_reason, "value") else str(exit_reason)
        if exit_val == ExitReason.AMS_3_STRIKES.value:
            return self._deterministic_fallback(transcript, visual_signs, scenario, skin_status, exit_reason)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an Emergency Department Triage Officer operating under ESI v4 standards.
Classify the patient into an ESI Level: 1 (Resuscitation), 2 (Emergent), 3 (Urgent), 4 (Less Urgent), or 5 (Non-urgent).
Return ONLY a valid JSON object matching this schema:
{{
  "esi_level": <int 1-5>,
  "urgency": "<Resuscitation|Emergent|Urgent|Less Urgent|Non-urgent>",
  "scenario_type": "{scenario['scenario_type']}",
  "subjective": "<patient stated symptoms>",
  "objective": "<observed visual postures and skin assessment>",
  "assessment": "<clinical nursing impression>",
  "plan": "<immediate triage action plan>",
  "audit_mode": "LLM_INFERENCE_VALIDATED"
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
[CUMULATIVE PATIENT TRANSCRIPT]: "{transcript}"
[PERCEPTION TELEMETRY OVER ENCOUNTER]:
- Chest Pain (Levine's sign): {visual_signs.get('chest_clutching', False)}
- Abdominal Guarding: {visual_signs.get('abdominal_clutching', False)}
- Cranial Distress (Headache): {visual_signs.get('head_clutching', False)}
- Facial Skin Status (CNN): {skin_status}
[SCENARIO CROSS-VALIDATION]: {scenario['scenario_type']} ({scenario['clinical_flag']})
[ENCOUNTER EXIT REASON]: {exit_val}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 256},
            "keep_alive": -1
        }

        try:
            start_t = time.time()
            res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TIMEOUT)
            if res.status_code == 200:
                out = json.loads(res.json().get("response", "{}"))
                out["scenario_details"] = scenario
                out["inference_latency_sec"] = round(time.time() - start_t, 2)
                return out
            else:
                return self._deterministic_fallback(transcript, visual_signs, scenario, skin_status, exit_reason)
        except Exception:
            return self._deterministic_fallback(transcript, visual_signs, scenario, skin_status, exit_reason)