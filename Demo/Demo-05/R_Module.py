"""
R_Module.py - Clinical Reasoning Engine (Step 4)
Scenario A/B/C Cross-Validation, 3-Strike Cognitive AMS Monitor, and Holistic SOAP Generation.
"""

import json
import time
import requests
from config import SystemConfig, ExitReason

class ReasoningEngine:
    def __init__(self):
        self.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"

    # -------------------------------------------------------------------------
    # 1. SCENARIO A / B / C CROSS-VALIDATION MATRIX
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_scenario(verbal_transcript: str, visual_signs: dict) -> dict:
        """
        Cross-validates physical gestures against verbal complaints:
        - Scenario A: Visual signs match verbal complaint (CONFIRMED)
        - Scenario B: Verbal complaint without physical sign (SUBJECTIVE_ONLY)
        - Scenario C: Physical distress sign detected while patient denies pain (CONFLICT / STOIC)
        """
        text = verbal_transcript.lower()
        chest_v = visual_signs.get("chest_clutching", False)
        abdo_v = visual_signs.get("abdominal_clutching", False)
        head_v = visual_signs.get("head_clutching", False)

        has_chest_vocal = any(k in text for k in ["chest", "heart", "เจ็บหน้าอก", "แน่นหน้าอก", "หัวใจ"])
        has_abdo_vocal = any(k in text for k in ["stomach", "abdomen", "belly", "ปวดท้อง", "เสียดท้อง", "จุกแน่น"])
        has_head_vocal = any(k in text for k in ["headache", "dizzy", "stroke", "ปวดหัว", "เวียนหัว", "มึนหัว", "ศีรษะ"])
        has_denial = any(k in text for k in ["fine", "nothing", "okay", "no pain", "ไม่เป็นไร", "สบายดี", "นิดหน่อย"])

        # Scenario A: Kinetic sign confirms verbal complaint
        if (chest_v and has_chest_vocal) or (abdo_v and has_abdo_vocal) or (head_v and has_head_vocal):
            return {
                "scenario_type": "SCENARIO_A_CONFIRMED",
                "clinical_flag": "HIGH_CONFIDENCE_ALIGNMENT",
                "notes": "Patient physical pain posturing directly corroborates stated complaint."
            }

        # Scenario C: Severe kinetic posturing observed despite verbal denial or silence
        if (chest_v or abdo_v) and (has_denial or len(text) == 0):
            return {
                "scenario_type": "SCENARIO_C_CLINICAL_CONFLICT",
                "clinical_flag": "SAFETY_OVERRIDE_STOIC_PATIENT",
                "notes": "Patient clutching vital pain zone while verbally minimizing symptoms. Stoic distress suspected."
            }

        # Scenario B: Subjective verbal complaint only (no kinetic clutching)
        return {
            "scenario_type": "SCENARIO_B_SUBJECTIVE_ONLY",
            "clinical_flag": "STANDARD_VERBAL_COMPLAINT",
            "notes": "Symptoms reported verbally without active kinetic clutching."
        }

    # -------------------------------------------------------------------------
    # 2. CONVERSATIONAL RELEVANCE & 3-STRIKE COGNITIVE MONITOR
    # -------------------------------------------------------------------------
    def check_relevance_and_pivot(self, utterance: str, current_strikes: int, lang: str = "th") -> tuple[bool, int, str]:
        """
        Evaluates patient coherence.
        Returns: (is_off_topic, updated_strikes, pivot_text_for_gTTS).
        """
        text = utterance.strip().lower()
        if len(text) == 0:
            return False, current_strikes, ""

        # Medical intent keywords
        medical_keywords = [
            "ปวด", "เจ็บ", "แน่น", "ไข้", "หนาว", "เวียน", "อ้วก", "ท้อง", "หัว", "แผล", "หายใจ", "ยา", "หมอ",
            "pain", "ache", "fever", "chest", "head", "stomach", "dizzy", "breath", "sick", "wound", "doctor"
        ]

        is_relevant = any(kw in text for kw in medical_keywords) or len(text.split()) >= 4

        # If answer is coherent and on-topic -> reset/hold strike
        if is_relevant:
            return False, current_strikes, ""

        # Off-topic detected -> Increment strike
        new_strikes = current_strikes + 1
        print(f"[!] Conversational Incoherence Detected: Strike {new_strikes}/{SystemConfig.MAX_OFFTOPIC_STRIKES}")

        # Strike 3: Altered Mental Status (AMS) triggered
        if new_strikes >= SystemConfig.MAX_OFFTOPIC_STRIKES:
            pivot_text = SystemConfig.SCRIPT_TEXTS[lang]["ams_alert"]
            return True, new_strikes, pivot_text

        # Strikes 1 & 2: Generate brief, Zero-PII steering pivot
        if lang == "th":
            pivot_text = "เข้าใจแล้วครับ แต่ตอนนี้ขอรบกวนช่วยบอกอาการไม่สบายหลักๆ ให้ผมทราบก่อนนะครับ"
        else:
            pivot_text = "I understand. However, please focus on your primary symptoms so I can assist you."

        return True, new_strikes, pivot_text

    # -------------------------------------------------------------------------
    # 3. MID-SESSION ACUTE PREEMPTION DETECTOR
    # -------------------------------------------------------------------------
    @staticmethod
    def check_acute_preemption(utterance: str, visual_signs: dict) -> tuple[bool, int]:
        """
        Checks for sudden mid-interview life-threatening decompensation.
        Returns (is_preempted, emergency_esi_level).
        """
        text = utterance.lower()
        chest_v = visual_signs.get("chest_clutching", False)

        # Critical Red-Flag Utterances
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
    # 4. DETERMINISTIC EMERGENCY SAFETY FALLBACK
    # -------------------------------------------------------------------------
    @staticmethod
    def _deterministic_fallback(transcript: str, visual_signs: dict, scenario: dict, skin_status: str, exit_reason: ExitReason) -> dict:
        """
        Immediate rule-based clinical classification based on ESI v4 standards.
        Fires if Ollama times out or drops network connection.
        """
        sc_type = scenario.get("scenario_type")

        # Fallback Logic Branch 1: Altered Mental Status (3-Strike Cognitive Failure)
        if exit_reason == ExitReason.AMS_3_STRIKES:
            return {
                "esi_level": 2,
                "urgency": "Emergent / High Risk",
                "scenario_type": sc_type,
                "subjective": f"Patient exhibiting progressive conversational confusion: \"{transcript}\"",
                "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}, Strike Count: 3",
                "assessment": "Suspected acute Altered Mental Status (AMS). High risk for neurological or metabolic deficit.",
                "plan": "Immediate nursing evaluation at kiosk, measure blood glucose, stroke screen (FAST protocol).",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

        # Fallback Logic Branch 2: High-Risk Cardiac / Acute Distress
        if sc_type in ["SCENARIO_A_CONFIRMED", "SCENARIO_C_CLINICAL_CONFLICT"] or visual_signs.get("chest_clutching", False):
            return {
                "esi_level": 2,
                "urgency": "Emergent / High Risk",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "Suspected acute coronary syndrome or critical thoracic distress.",
                "plan": "Immediate 12-lead ECG, priority transfer to resuscitation/monitored bed.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

        # Fallback Logic Branch 3: Acute Abdomen
        if visual_signs.get("abdominal_clutching", False):
            return {
                "esi_level": 3,
                "urgency": "Urgent",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "Acute abdominal pain with physical guarding, multiple diagnostic resources needed.",
                "plan": "Transfer to acute care bay, establish IV access, order abdominal labs.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

        # Fallback Logic Branch 4: Baseline Stable Presentation
        return {
            "esi_level": 4,
            "urgency": "Less Urgent",
            "scenario_type": sc_type,
            "subjective": transcript,
            "objective": f"Visual signs: {visual_signs}, Facial Erythema: {skin_status}",
            "assessment": "Stable ambulatory presentation. Single resource anticipated.",
            "plan": "Standard outpatient triage queue.",
            "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
        }

    # -------------------------------------------------------------------------
    # 5. HOLISTIC FINAL CLINICAL SYNTHESIS
    # -------------------------------------------------------------------------
    def evaluate_final_case(self, conversation_history: list, visual_signs: dict, skin_status: str, exit_reason: ExitReason, language: str) -> dict:
        """
        Synthesizes the entire multi-turn clinical encounter into a definitive ESI Level + SOAP note.
        """
        transcript = " ".join([t["text"] for t in conversation_history if t.get("role") == "patient"])
        scenario = self.evaluate_scenario(transcript, visual_signs)

        # Force ESI 2 immediately if exit reason is Altered Mental Status
        if exit_reason == ExitReason.AMS_3_STRIKES:
            return self._deterministic_fallback(transcript, visual_signs, scenario, skin_status, exit_reason)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Clinical Triage Officer following the Emergency Severity Index (ESI v4).
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