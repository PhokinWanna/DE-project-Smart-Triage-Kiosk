import requests
# import json
import time

# ตั้งค่า Ollama API แบบ Local
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2" 


# system
def analyze_triage(transcript, visual_flag):
    system_prompt = """
[SYSTEM PROMPT]
You are an expert Emergency Triage Assistant. 
Rule 1: NO DIAGNOSIS. Do not guess or name the disease.
Rule 2: Classify urgency strictly based on the Emergency Severity Index (ESI 1-5), That is Level 1 = (Resuscitation)
Level 2 =(Emergent)
Level 3 =(Urgent)
Level 4 =(Less Urgent)
and Level 5 =(Non-Urgent).
Rule 3: Analyze both the patient's spoken transcript and the visual physical signs provided.
Rule 4: Output ONLY in valid JSON format containing 'esi_level' and 'clinical_summary'.

[EXPECTED OUTPUT FORMAT]
{
  "esi_level": <integer 1-5>,
  "clinical_summary": "<string explaining the reasoning based on both text and visual signs>"
}
"""

    user_prompt = f"""
[INPUT DATA]
Transcript: "{transcript}"
Visual_Flag: "{visual_flag}"
"""

    full_prompt = system_prompt + user_prompt

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    print(f"\nProcessing Multimodal Data...")
    print(f" -> Text: '{transcript}'")
    print(f" -> Vision: {visual_flag}")
    
    start_time = time.time()
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        
        end_time = time.time()
        print(f"Time Taken: {end_time - start_time:.2f} seconds\n")
        
        print("------ TRIAGE RESULT ------")
        print(result)
        print("----------------------------")
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running (ollama serve).")


if __name__ == "__main__":
    print("--- Smart Triage LLM ---")
    
    # Scenario 1 = อาการปวดหัวธรรมดา ไม่มีท่าทางผิดปกติ expect= ESI 4,5
    analyze_triage(
        transcript="I have a headache and feel a bit tired since this morning.", 
        visual_flag="Normal Posture"
    )
    time.sleep(2)

    # Scenario 2 = อาการปวดหัว แต่ภาพจับได้ว่ากุมขมับ expect= ESI สูงขึ้น อาจจะ 3 หรือ 2 ขึ้นอยู่กับคำพูด
    analyze_triage(
        transcript="My head hurts a lot right now.", 
        visual_flag="HEADACHE CONFIRMED (Patient clutching head)"
    )
    time.sleep(2)
    
    # Scenario 3 = อาการเจ็บหน้าอกแบบเงียบๆ expect= คาดหวัง ESI 1,2
    analyze_triage(
        transcript="...",  # คนไข้เงียบ หรือพูดไม่ออก
        visual_flag="CHEST PAIN CONFIRMED (Patient clutching chest tightly)"
    )