from transformers import pipeline

# โหลดโมเดลภาษาไทยที่รองรับการจำแนกข้อความ (Text Classification)
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

# ฟังก์ชันวิเคราะห์ข้อความ
def classify_accident(text):
    result = classifier(text)
    label = result[0]["label"]  # ดึงค่าที่โมเดลจำแนกได้

    # Rule-Based สำหรับกรณีที่แน่ชัด
    if "ชนท้าย" in text:
        return "คู่กรณีผิด"
    elif "ย้อนศร" in text or "ฝ่าไฟแดง" in text:
        return "คุณผิด"
    elif label == "positive":
        return "โอกาสชนะสูง"
    else:
        return "กรุณาให้ข้อมูลเพิ่มเติม"

# ทดสอบระบบ
user_input = "ฉันโดนรถชนท้าย"
bot_response = classify_accident(user_input)
print("Chatbot:", bot_response)
