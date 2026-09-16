"""
R_Module.py - Clinical Reasoning Engine
Evaluates Scenario A/B/C logic, constructs SOAP notes, and assigns ESI Levels (1-5).
"""

import json
import time
import requests
from config import SystemConfig

class ReasoningEngine:
    def __init__(self):
        self.endpoint = f"{SystemConfig.OLLAMA_HOST}/api/generate"

    @staticmethod
    def evaluate_scenario(verbal_complaint: str, visual_signs: dict) -> dict:
        """
        Reconstructs Demo-03 Scenario Matrix:
        - Scenario A: Visual signs match verbal complaint (CONFIRMED)
        - Scenario B: Verbal complaint without physical signs (SUBJECTIVE_ONLY)
        - Scenario C: High visual distress detected while patient denies/minimizes verbally (CONFLICT / STOIC)
        """
        text = verbal_complaint.lower()
        chest_visual = visual_signs.get("chest_clutching", False)
        abdo_visual = visual_signs.get("abdominal_clutching", False)
        head_visual = visual_signs.get("head_clutching", False)

        has_chest_verbal = any(k in text for k in ["chest", "heart", "เจ็บหน้าอก", "แน่นหน้าอก", "หัวใจ"])
        has_abdo_verbal = any(k in text for k in ["stomach", "abdomen", "belly", "ปวดท้อง", "เสียดท้อง"])
        has_head_verbal = any(k in text for k in ["headache", "dizzy", "stroke", "ปวดหัว", "เวียนหัว", "ศีรษะ"])
        has_denial = any(k in text for k in ["fine", "nothing", "okay", "ไม่เป็นไร", "สบายนิดหน่อย", "นิดหน่อย"])

        # Scenario A: Mutual confirmation
        if (chest_visual and has_chest_verbal) or (abdo_visual and has_abdo_verbal) or (head_visual and has_head_verbal):
            return {
                "scenario_type": "SCENARIO_A_CONFIRMED",
                "clinical_flag": "HIGH_CONFIDENCE_ALIGNMENT",
                "notes": "Patient physical pain posturing directly matches verbal complaint."
            }

        # Scenario C: Visual distress with verbal denial/silence (Silent acute emergency)
        if (chest_visual or abdo_visual) and (has_denial or len(text) == 0):
            return {
                "scenario_type": "SCENARIO_C_CLINICAL_CONFLICT",
                "clinical_flag": "SAFETY_OVERRIDE_STOIC_PATIENT",
                "notes": "Patient clutching vital pain zone while denying symptoms. Severe distress suspected."
            }

        # Scenario B: Subjective verbal complaint only
        return {
            "scenario_type": "SCENARIO_B_SUBJECTIVE_ONLY",
            "clinical_flag": "STANDARD_VERBAL_COMPLAINT",
            "notes": "Symptoms reported verbally without active kinetic clutching."
        }

    @staticmethod
    def _deterministic_fallback(transcript: str, visual_signs: dict, scenario: dict, skin_status: str) -> dict:
        """Immediate rule-based clinical fallback if LLM stalls or fails."""
        sc_type = scenario.get("scenario_type")

        if sc_type in ["SCENARIO_A_CONFIRMED", "SCENARIO_C_CLINICAL_CONFLICT"] or visual_signs.get("chest_clutching", False):
            return {
                "esi_level": 2,
                "urgency": "Emergent / High Risk",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual Signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "High probability acute coronary syndrome or critical distress.",
                "plan": "Immediate 12-lead ECG, priority transfer to resuscitation/monitored bed.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }
        elif visual_signs.get("abdominal_clutching", False):
            return {
                "esi_level": 3,
                "urgency": "Urgent",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual Signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "Acute abdominal pain, multi-resource evaluation required.",
                "plan": "Assign to acute care bay, prepare IV access, order routine blood panel.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }
        else:
            return {
                "esi_level": 4,
                "urgency": "Less Urgent",
                "scenario_type": sc_type,
                "subjective": transcript,
                "objective": f"Visual Signs: {visual_signs}, Facial Erythema: {skin_status}",
                "assessment": "Stable presentation, single resource anticipated.",
                "plan": "Standard outpatient triage queue.",
                "audit_mode": "DETERMINISTIC_FALLBACK_ACTIVE"
            }

    def evaluate_case(self, conversation_history: list, visual_signs: dict, skin_status: str, language: str) -> dict:
        """
        Runs Scenario Cross-Verification and prompts Llama 3.2 for an ESI + SOAP package.
        """
        transcript = " ".join([t["text"] for t in conversation_history if t.get("role") == "patient"])
        scenario = self.evaluate_scenario(transcript, visual_signs)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Clinical Triage Officer following Emergency Severity Index (ESI v4).
Classify the patient into ESI Level 1 (Resuscitation), 2 (Emergent), 3 (Urgent), 4 (Less Urgent), or 5 (Non-urgent).
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
[PATIENT TRANSCRIPT]: "{transcript}"
[VISUAL TELEMETRY]:
- Chest Pain (Levine's sign): {visual_signs.get('chest_clutching', False)}
- Abdominal Guarding: {visual_signs.get('abdominal_clutching', False)}
- Cranial Distress (Headache): {visual_signs.get('head_clutching', False)}
- Facial Skin Assessment (CNN): {skin_status}
[CROSS-VALIDATION RESULT]: {scenario['scenario_type']} ({scenario['clinical_flag']})
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
            res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.OLLAMA_TIMEOUT)
            if res.status_code == 200:
                out = json.loads(res.json().get("response", "{}"))
                out["scenario_details"] = scenario
                return out
            else:
                return self._deterministic_fallback(transcript, visual_signs, scenario, skin_status)
        except Exception:
            return self._deterministic_fallback(transcript, visual_signs, scenario, skin_status)