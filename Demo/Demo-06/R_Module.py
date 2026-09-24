"""
R_Module.py - Clinical Reasoning Engine (Step 4)
Dynamic OPQRST Probing, Scenario A/B/C Matrix, 3-Strike Cognitive AMS, and SOAP Synthesis.
"""

import json
import time
import requests
from config import SystemConfig, ExitReason

class ReasoningEngine:
    def __init__(self):
        self.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"
        self.warmup() # Burn-in during boot so patient #1 experiences 0s delay
        
# -------------------------------------------------------------------
    def warmup(self):
        """Pins Llama 3.2 in VRAM and burns in CUDA kernels."""
        print("[*] Pre-warming Llama 3.2 in VRAM (Pre-Flight Test)...")
        payload = {
            "model": SystemConfig.OLLAMA_MODEL,
            "prompt": "ok",
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": -1 # Instructs Ollama to keep weights in VRAM permanently
        }
        try:
            requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TIMEOUT)
            print("[+] Llama 3.2 is hot and pinned in VRAM.")
        except Exception as e:
            print(f"[!] Warmup Notice: Ollama not reachable on boot: {e}")
# ------------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # 1. STOPPING INTENT DETECTOR
    # -------------------------------------------------------------------------
    @staticmethod
    def is_stopping_phrase(utterance: str, lang: str = "th") -> bool:
        """
        Detects if the patient explicitly indicates they are finished explaining.
        e.g., 'แค่นี้ครับ', 'หมดแล้วค่ะ', 'that's all', 'nothing else'.
        """
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
        """
        Scans for sudden critical deterioration mid-encounter.
        Returns: (is_preempted, emergency_esi_level)
        """
        text = utterance.lower()
        chest_v = visual_signs.get("chest_clutching", False)

        # Critical Red-Flag Tokens
        critical_words = [
            "จะตาย", "หายใจไม่ออก", "หมดสติ", "วูบ", "แน่นหน้าอกมาก", "หัวใจจะหยุด",
            "cannot breathe", "passing out", "crushing chest", "heart attack", "unconscious"
        ]
        has_critical_vocal = any(w in text for w in critical_words)

        # ESI 1: Immediate Resuscitation Trigger
        if "หมดสติ" in text or "unconscious" in text or "cannot breathe" in text:
            return True, 1

        # ESI 2: High-Risk Acute Preemption (Levine's sign or crushing chest pain)
        if chest_v or has_critical_vocal:
            return True, 2

        return False, 0

    # -------------------------------------------------------------------------
    # 3. DYNAMIC OPQRST CLINICAL PROBING & RELEVANCE MONITOR
    # -------------------------------------------------------------------------
    def evaluate_turn_and_probe(self, utterance: str, history: list, current_strikes: int, lang: str = "th") -> dict:
        """
        Evaluates a patient's turn:
        - If off-topic -> Increments strikes; Strike 3 triggers AMS.
        - If on-topic -> Determines if key OPQRST details are missing and generates
                         ONE short clinical question (<=25 tokens, Zero-PII, no disease labels).
        """
        text = utterance.strip().lower()

        # Check Medical Intent Keywords
        medical_keywords = [
            "ปวด", "เจ็บ", "แน่น", "ไข้", "หนาว", "เวียน", "อ้วก", "ท้อง", "หัว", "แผล", "หายใจ", "ยา", "หมอ",
            "pain", "ache", "fever", "chest", "head", "stomach", "dizzy", "breath", "sick", "wound", "doctor"
        ]
        is_relevant = any(kw in text for kw in medical_keywords) or len(text.split()) >= 4

        # Case A: Off-Topic / Confusion Detected
        if not is_relevant and len(text) > 0:
            new_strikes = current_strikes + 1
            print(f"[!] Off-Topic Utterance Detected. Strike {new_strikes}/{SystemConfig.MAX_OFFTOPIC_STRIKES}")

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
            else:
                pivot_text = "เข้าใจแล้วค่ะ แต่ตอนนี้รบกวนช่วยบอกอาการเจ็บป่วยหลักๆ ให้ดิฉันทราบก่อนนะคะ" if lang == "th" else \
                             "I understand. However, please tell me your main physical symptoms first."
                return {
                    "is_off_topic": True,
                    "strikes": new_strikes,
                    "ams_triggered": False,
                    "action": "STEER_PIVOT",
                    "speech_output": pivot_text,
                    "use_template": False
                }

        # Case B: On-Topic Utterance -> Evaluate Clinical Completeness (OPQRST)
        conversation_context = " ".join([turn.get("text", "") for turn in history if turn.get("role") == "patient"])
        combined_text = f"{conversation_context} {utterance}".strip()

        # Check if basic OPQRST descriptors are already provided
        has_timing = any(w in combined_text for w in ["เมื่อวาน", "กี่โมง", "วัน", "ชั่วโมง", "นาที", "yesterday", "hours", "days", "since"])
        has_severity = any(w in combined_text for w in ["มาก", "น้อย", "เต็ม 10", "10", "severe", "mild", "moderate", "scale"])

        # If details are already comprehensive, signal that active listening nudge or confirmation is appropriate
        if has_timing and has_severity:
            return {
                "is_off_topic": False,
                "strikes": current_strikes,
                "ams_triggered": False,
                "action": "DETAILS_SUFFICIENT",
                "speech_output": "",
                "use_template": True
            }

        # Targeted Probing via Llama 3.2 (Capped at PROBE_MAX_TOKENS for sub-350ms latency)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an Emergency Triage Nurse asking ONE brief follow-up question.
STRICT RULES:
1. Ask ONLY about missing symptom descriptors: onset time, pain severity (1-10 scale), or sensation type.
2. DO NOT state or guess any disease names or diagnoses.
3. DO NOT include any patient personal data or names.
4. Output EXACTLY ONE short question under 15 words in { 'Thai' if lang == 'th' else 'English' }.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Patient stated: "{combined_text}"
Generate the single next triage question:
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
                # Clean any stray formatting
                probe_q = probe_q.replace('"', '').replace('\n', ' ')
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

        # Rule-based fallback probe if LLM is slow
        fallback_probe = "เป็นมานานกี่ชั่วโมงแล้วคะ และเจ็บมากไหมคะ เต็ม 10 ให้เท่าไหร่คะ?" if lang == "th" else \
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
        """
        Cross-validates kinetic postures against stated complaints.
        """
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

        # Scenario C: Visual distress with verbal denial/silence (Silent acute emergency)
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
        """
        Hardcoded clinical triage rules based on ESI v4 standards.
        Fires if Ollama is unavailable or times out (>5.0s).
        """
        sc_type = scenario.get("scenario_type")

        # 1. Altered Mental Status (3-Strike Cognitive Failure)
        if exit_reason == ExitReason.AMS_3_STRIKES:
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

        # 2. Critical Cardiac / Severe Distress
        if exit_reason == ExitReason.EMERGENCY_INTERRUPT or sc_type in ["SCENARIO_A_CONFIRMED", "SCENARIO_C_CLINICAL_CONFLICT"] or visual_signs.get("chest_clutching", False):
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

        # 3. Acute Abdomen
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

        # 4. Baseline Stable Presentation
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
    # 6. FINAL CLINICAL SYNTHESIS (FULL ENCOUNTER EVALUATION)
    # -------------------------------------------------------------------------
    def evaluate_final_case(self, conversation_history: list, visual_signs: dict, skin_status: str, exit_reason: ExitReason, language: str) -> dict:
        """
        Synthesizes the complete patient timeline into the final ESI Level + SOAP note.
        """
        transcript = " ".join([t["text"] for t in conversation_history if t.get("role") == "patient"])
        scenario = self.evaluate_scenario(transcript, visual_signs)

        # Immediate rule assignment for cognitive failure
        if exit_reason == ExitReason.AMS_3_STRIKES:
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
[ENCOUNTER EXIT REASON]: {exit_reason.value}
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