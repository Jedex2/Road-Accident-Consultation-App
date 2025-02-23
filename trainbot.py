from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ตัวอย่างข้อความสำหรับฝึกโมเดล (X = ข้อความ, y = หมวดหมู่)
train_texts = [
    "ฉันโดนรถชนท้าย",  
    "ฉันขับรถชนคนข้ามถนน",
    "รถของฉันถูกเฉี่ยว",
    "ฉันเป็นฝ่ายถูกต้อง",
    "ฉันอาจผิดเพราะเปลี่ยนเลนกระทันหัน"
    "ฉันชนคนตาย"
    "ฉันโดนรถชนข้างรถ"
]
train_labels = [
    "คู่กรณีผิด",  # ข้อความ 1 → คู่กรณีผิด
    "คุณอาจมีความผิด",  # ข้อความ 2 → ผู้ขับขี่อาจผิด
    "อุบัติเหตุเล็กน้อย",  # ข้อความ 3 → อุบัติเหตุเล็กน้อย
    "คุณชนะคดี",  # ข้อความ 4 → ผู้ขับขี่ถูก
    "คุณอาจแพ้คดี"  # ข้อความ 5 → ผู้ขับขี่อาจแพ้คดี
]

# Tokenizer แบบภาษาไทย
def tokenize_thai(text):
    return word_tokenize(text, engine="newmm")

# สร้าง Pipeline สำหรับทำ Text Classification
model = make_pipeline(CountVectorizer(tokenizer=tokenize_thai), MultinomialNB())

# เทรนโมเดล
model.fit(train_texts, train_labels)

# ทดสอบ Chatbot
def chatbot_response(text):
    return model.predict([text])[0]  # ให้โมเดลทำนายหมวดหมู่

# ตัวอย่างการใช้งาน
user_input = "ฉันโดนรถชนข้างรถ"
response = chatbot_response(user_input)
print("Chatbot:", response)
