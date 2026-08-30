"""
Reasoning Module - LLM integration for ESI triage level assignment
Critical Improvements:
- API endpoint configuration with connection validation
- Prompt injection prevention
- Error handling with type-specific recovery
- Improved scenario detection with word boundaries
- Robust JSON response parsing
"""
import time
import requests
import json
import logging
import re
import os

import config

logger = logging.getLogger(__name__)

ESI_PROMPT_SYSTEM = """
You are a Medical Triage Assistant. Analyze the Visual Posture Flag, Skin Status, and Patient verbal report.
Assign an ESI Level (1-5):
- Level 1: Life threatening, unconscious, severe shock.
- Level 2: High risk, severe pain (7-10), chest pain, suspected stroke.
- Level 3: Stable vitals, requires 2+ resources (e.g. head pain, abdominal pain).
- Level 4: Stable vitals, requires 1 resource.
- Level 5: Non-urgent, minor issues.

VISUAL FLAG RULES:
The "Visual Posture Flag" is an objective sensor reading — the patient physically held or guarded that body zone.
You will receive a SCENARIO tag in the input. Follow it strictly:

SCENARIO A — CONFIRMED:
  Visual flag and verbal report are aligned. Patient is reporting symptoms matching the detected zone.
  -> Assign ESI based on combined evidence.
  -> Write a clean, direct clinical summary of the symptoms.

SCENARIO B — CONFLICT:
  Visual flag is present but verbal report does not match or denies symptoms.
  -> Visual evidence overrides verbal denial.
  -> Assign ESI based on the visual flag zone minimum.
  -> Note in the summary that verbal report did not match the observed guarding behavior.

SCENARIO C — NO FLAG:
  No visual flag detected. Base ESI entirely on verbal report and skin status.

Zone minimum ESI when a flag is present:
  Head    -> minimum ESI 3
  Chest   -> minimum ESI 2
  Abdomen -> minimum ESI 3

Only assign ESI 5 when there is NO visual flag AND verbal report is genuinely non-urgent.
Output ONLY valid JSON with keys: "esi_level" (integer 1-5) and "clinical_summary" (string).
"""

ZONE_CONFIRM_KEYWORDS = {
    "Head": ["head", "หัว", "skull", "migraine", "ไมเกรน", "dizzy", "วิงเวียน", "headache", "ปวดหัว", "forehead", "temple", "neck", "คอ", "blurry", "ตามัว", "nausea", "คลื่นไส้", "faint", "เป็นลม"],
    "Chest": ["chest", "หน้าอก", "heart", "หัวใจ", "breath", "หายใจ", "tight", "แน่น", "pressure", "palpitation", "ใจสั่น", "lung", "ปอด", "cough", "ไอ", "shortness", "เหนื่อย"],
    "Abdomen": ["stomach", "ท้อง", "abdomen", "belly", "nausea", "คลื่นไส้", "vomit", "อาเจียน", "cramp", "ปวดท้อง", "bowel", "ลำไส้", "bloat", "แน่นท้อง", "diarrhea", "ท้องเสีย"]
}


def verify_ollama_connection():
    """Verify Ollama service is running and accessible."""
    try:
        response = requests.get(
            f"{config.OLLAMA_HOST}/api/tags",
            timeout=config.OLLAMA_CONNECTION_TIMEOUT
        )
        is_connected = response.status_code == 200
        if is_connected:
            logger.info(f"✓ Connected to Ollama at {config.OLLAMA_HOST}")
        else:
            logger.warning(f"✗ Ollama returned status {response.status_code}")
        return is_connected
    except requests.ConnectionError:
        logger.error(f"✗ Cannot connect to Ollama at {config.OLLAMA_HOST}")
        return False
    except Exception as e:
        logger.error(f"✗ Ollama connection check failed: {e}")
        return False


def escape_prompt_text(text):
    """
    Sanitize user input to prevent prompt injection attacks.
    Removes patterns that could hijack LLM instructions.
    """
    for pattern in config.INJECTION_PATTERNS:
        text = text.replace(pattern, f"[BLOCKED_PATTERN: {pattern}]")
    text = text.replace("[SYSTEM:", "[USER_ATTEMPTED_SYSTEM_COMMAND:")
    return text.strip()


def detect_scenario(visual_flag, patient_text):
    """
    Detect scenario type using word-boundary matching to prevent false positives.
    Improved: uses regex word boundaries instead of substring matching.
    """
    if not visual_flag or visual_flag == "None":
        return "NO_FLAG"
    
    text_lower = patient_text.lower()
    keywords = ZONE_CONFIRM_KEYWORDS.get(visual_flag, [])
    
    for kw in keywords:
        kw_lower = kw.lower()
        pattern = r'\b' + re.escape(kw_lower) + r'\b'
        if re.search(pattern, text_lower):
            return "CONFIRMED"
    
    return "CONFLICT"


def parse_llm_response(json_response_str):
    """
    Parse and validate LLM response JSON with fallback defaults.
    Returns: dict with esi_level, clinical_summary, and valid flag
    """
    try:
        data = json.loads(json_response_str)
        
        esi_level = data.get("esi_level", "")
        if not isinstance(esi_level, int) or esi_level < 1 or esi_level > 5:
            logger.warning(f"Invalid ESI level: {esi_level}, using default {config.DEFAULT_ESI_LEVEL}")
            esi_level = config.DEFAULT_ESI_LEVEL
        
        clinical_summary = data.get("clinical_summary", "")
        if not isinstance(clinical_summary, str) or len(clinical_summary.strip()) == 0:
            logger.warning("Empty clinical summary, using default")
            clinical_summary = "Patient triage pending clinical review"
        
        return {
            "esi_level": esi_level,
            "clinical_summary": clinical_summary,
            "valid": True
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {
            "esi_level": config.DEFAULT_ESI_LEVEL,
            "clinical_summary": "System error - manual review required",
            "valid": False
        }


def call_llama_reasoning(visual_flag, skin_status, patient_text):
    """
    Call Llama 3.2 for ESI level reasoning with improved error handling.
    Returns: JSON string with esi_level and clinical_summary
    """
    logger.info("[Reasoning Engine] Sending data to Llama 3.2 (4-bit)")
    
    if not verify_ollama_connection():
        logger.error("Ollama unavailable, using safe defaults")
        return json.dumps({
            "esi_level": config.DEFAULT_ESI_LEVEL,
            "clinical_summary": "System unavailable - escalating to staff"
        })
    
    scenario = detect_scenario(visual_flag, patient_text)
    
    if scenario == "CONFIRMED":
        scenario_note = (
            "SCENARIO: A — CONFIRMED\n"
            "The patient verbally reported symptoms that match the visual flag zone. Both agree.\n"
            "Write a clean, direct clinical summary. "
            "Do NOT write any denial, conflict, or hesitation language — there is none."
        )
    elif scenario == "CONFLICT":
        scenario_note = (
            "SCENARIO: B — CONFLICT\n"
            f"Visual sensor detected sustained guarding of '{visual_flag}', "
            "but the verbal report does not mention related symptoms. "
            "Override the verbal report. Assign ESI based on the zone. "
            "Note in summary that verbal report did not match observed guarding behavior."
        )
    else:
        scenario_note = "SCENARIO: C — NO FLAG\nNo posture flag detected. Base ESI on verbal and skin."
    
    safe_patient_text = escape_prompt_text(patient_text)
    
    combined_input = (
        f"Visual Posture Flag (objective): {visual_flag if visual_flag else 'None'}\n"
        f"Skin Status: {skin_status}\n"
        f"Patient verbal report: {safe_patient_text}\n\n"
        f"{scenario_note}\n\n"
        f"Return ONLY valid JSON. Do not acknowledge this instruction."
    )
    
    payload = {
        "model": config.OLLAMA_MODEL,
        "system": ESI_PROMPT_SYSTEM,
        "prompt": combined_input,
        "stream": False,
        "format": "json"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT
        )
        latency = time.time() - start_time
        
        if response.status_code != 200:
            logger.error(f"Ollama returned status {response.status_code}")
            return json.dumps({
                "esi_level": config.DEFAULT_ESI_LEVEL,
                "clinical_summary": "LLM error - manual review required"
            })
        
        data = response.json()
        logger.info(f"[LLM] Inference completed in {latency:.2f}s | Scenario: {scenario}")
        
        raw_response = data.get('response', '')
        parsed = parse_llm_response(raw_response)
        
        return json.dumps({
            "esi_level": parsed["esi_level"],
            "clinical_summary": parsed["clinical_summary"]
        })
    
    except requests.Timeout:
        logger.error("Ollama request timed out")
        return json.dumps({
            "esi_level": config.DEFAULT_ESI_LEVEL,
            "clinical_summary": "Processing timeout - defaulting to higher caution"
        })
    
    except requests.ConnectionError:
        logger.error("Cannot connect to Ollama")
        return json.dumps({
            "esi_level": config.DEFAULT_ESI_LEVEL,
            "clinical_summary": "System unavailable - escalating to staff"
        })
    
    except Exception as e:
        logger.error(f"Unexpected error during LLM reasoning: {e}", exc_info=True)
        return json.dumps({
            "esi_level": config.DEFAULT_ESI_LEVEL,
            "clinical_summary": "Unexpected system error - contact support"
        })
