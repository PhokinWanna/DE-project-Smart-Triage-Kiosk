# test 1
# import ollama

# # กำหนดชื่อโมเดลที่เราโหลดมา (เช็คให้ชัวร์ว่าชื่อตรงกับที่เรา pull มา)
# MODEL_NAME = "llama3.2"

# def think_and_reply(prompt):
#     print("⏳ สมองกำลังคิด... (รอแป๊บ)")
    
#     # ส่งข้อความไปหา Ollama
#     response = ollama.chat(model=MODEL_NAME, messages=[
#         {
#             'role': 'user',
#             'content': prompt,
#         },
#     ])
    
#     # ดึงเฉพาะเนื้อหาคำตอบออกมา
#     return response['message']['content']

# # --- ส่วนของการทดสอบ ---
# if __name__ == "__main__":
#     # จำลองสถานการณ์: ส่งข้อมูลอาการคนไข้เข้าไป
#     user_input = "ผู้ป่วยชาย อายุ 50 ปี มือกุมหน้าอกข้างซ้าย เหงื่อออกเยอะมาก และหายใจไม่ออก ช่วยสรุปอาการและประเมินความเสี่ยงหน่อย"
    
#     print(f"🏥 Input: {user_input}")
#     print("-" * 30)
    
#     # เรียกใช้ฟังก์ชัน
#     ai_reply = think_and_reply(user_input)
    
#     print(f"🤖 AI Answer:\n{ai_reply}")
#     print("-" * 30)

# ------------------------------------------------------------------------------------------------------------


import ollama
import json # เพิ่ม library นี้มาช่วยดูดข้อมูล

MODEL_NAME = "llama3.2"

def think_and_reply(prompt):
    print("⏳ สมองกำลังคิด... (รอแป๊บ)")
    
    # --- เคล็ดลับวิชา: System Prompt ---
    # นี่คือการสะกดจิต AI ให้รู้หน้าที่ของตัวเอง
    system_instruction = """
    คุณคือพยาบาลคัดกรองอัจฉริยะ หน้าที่คือรับข้อมูลอาการและแปลงเป็น JSON format เท่านั้น
    ห้ามพูดพร่ำเพรื่อ ห้ามแนะนำการรักษา ห้ามวินิจฉัยโรค
    
    ให้ตอบกลับเป็น JSON โครงสร้างนี้เท่านั้น:
    {
        "summary": "สรุปอาการสั้นๆ ไม่เกิน 10 คำ",
        "pain_level": "ระดับความเจ็บปวด 1-10 (ประเมินจากข้อความ)",
        "urgency": "ระดับความเร่งด่วน (ต่ำ/ปานกลาง/สูง/ฉุกเฉิน)",
        "department": "แผนกที่ควรส่งต่อ (เช่น อายุรกรรม, ศัลยกรรม, หัวใจ)"
    }
    ตอบเป็นภาษาไทยใน value ของ JSON
    """

    response = ollama.chat(model=MODEL_NAME, messages=[
        {'role': 'system', 'content': system_instruction}, # สั่งจิตก่อน
        {'role': 'user', 'content': prompt}, # แล้วค่อยส่งอาการ
    ])
    
    return response['message']['content']

if __name__ == "__main__":
    # โจทย์เดิม
    # user_input = "ผู้ป่วยชาย อายุ 50 ปี มือกุมหน้าอกข้างซ้าย เหงื่อออกเยอะมาก และหายใจไม่ออก"
    user_input = ""
    print(f"🏥 Input: {user_input}")
    print("-" * 30)
    
    ai_reply = think_and_reply(user_input)
    
    print(f"🤖 AI Answer (Raw): {ai_reply}")
    print("-" * 30)
    
    # ลองแปลง String เป็น JSON Object จริงๆ เพื่อพิสูจน์ว่าเอาไปใช้ต่อได้
    try:
        data = json.loads(ai_reply)
        print("✅ แปลงเป็นตัวแปร Python สำเร็จ!")
        print(f"ระดับความเร่งด่วน: {data['urgency']}")
        print(f"ส่งแผนก: {data['department']}")
    except:
        print("❌ ผิด Format")