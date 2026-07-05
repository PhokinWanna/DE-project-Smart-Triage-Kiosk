import time
import requests

LLM_MODEL = "llama3.2:latest"

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

def detect_scenario(visual_flag, patient_text):
    if not visual_flag or visual_flag == "None":
        return "NO_FLAG"
    text_lower = patient_text.lower()
    keywords = ZONE_CONFIRM_KEYWORDS.get(visual_flag, [])
    for kw in keywords:
        if kw.lower() in text_lower:
            return "CONFIRMED"
    return "CONFLICT"

def call_llama_reasoning(visual_flag, skin_status, patient_text):
    print("\n[Reasoning Engine] Sending Data to Llama 3.2 (4-bit)")
    scenario = detect_scenario(visual_flag, patient_text)

    if scenario == "CONFIRMED":
        scenario_note = "SCENARIO: A — CONFIRMED\nThe patient verbally reported symptoms that match the visual flag zone. Both agree."
    elif scenario == "CONFLICT":
        scenario_note = "SCENARIO: B — CONFLICT\nVisual sensor detected sustained guarding, but verbal report does not match. Override verbal report."
    else:
        scenario_note = "SCENARIO: C — NO FLAG\nNo posture flag detected. Base ESI on verbal and skin."

    combined_input = (
        f"Visual Posture Flag (objective): {visual_flag if visual_flag else 'None'}\n"
        f"Skin Status: {skin_status}\n"
        f"Patient verbal report: {patient_text}\n\n"
        f"{scenario_note}"
    )

    payload = {
        "model": LLM_MODEL,
        "system": ESI_PROMPT_SYSTEM,
        "prompt": combined_input,
        "stream": False,
        "format": "json"
    }

    try:
        start_time = time.time()
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        latency = time.time() - start_time
        data = response.json()
        print(f"[Reasoning Engine] Inference Time : {latency:.2f} seconds | Scenario: {scenario}")
        return data.get('response', f"\n[Ollama API Error]: {data}\n")
    except Exception as e:
        return f'{{"error": "Error: {e}"}}'