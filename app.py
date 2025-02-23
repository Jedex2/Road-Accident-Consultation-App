from flask import Flask, render_template, request, jsonify
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle  # ใช้สำหรับบันทึกและโหลดโมเดล

app = Flask(__name__)

# -------------------------------
# โหลดโมเดล Naïve Bayes ที่ฝึกไว้
# -------------------------------
train_texts = [
    "ฉันโดนรถชนท้าย",  
    "ฉันขับรถชนคนข้ามถนน",
    "รถของฉันถูกเฉี่ยว",
    "ฉันเป็นฝ่ายถูกต้อง",
    "ฉันอาจผิดเพราะเปลี่ยนเลนกระทันหัน",
    "ฉันชนคนตาย",
    "ฉันโดนรถชนข้างรถ"
    "ฉันชนคนขี่มอเตอร์ไซต์ชื่อกอล์ฟเข้าห้องไอซียู"
]

train_labels = [
    "คู่กรณีผิด",  # ข้อความ 1 → คู่กรณีผิด
    "คุณอาจมีความผิด",  # ข้อความ 2 → ผู้ขับขี่อาจผิด
    "อุบัติเหตุเล็กน้อย",  # ข้อความ 3 → อุบัติเหตุเล็กน้อย
    "คุณชนะคดี",  # ข้อความ 4 → ผู้ขับขี่ถูก
    "คุณอาจแพ้คดี",  # ข้อความ 5 → ผู้ขับขี่อาจแพ้คดี
    "คุณอาจแพ้คดี",  # ข้อความ 6 → ผู้ขับขี่อาจแพ้คดี
    "คู่กรณีผิด"  # ข้อความ 7 → คู่กรณีผิด
    "คู่กรณีผิด แนะนําให้หนี ออกจากจุดเกิดเหตุ"  # ข้อความ 8 → คู่กรณีผิด
]

def tokenize_thai(text):
    return word_tokenize(text, engine="newmm")

# สร้างและเทรนโมเดล
model = make_pipeline(CountVectorizer(tokenizer=tokenize_thai), MultinomialNB())
model.fit(train_texts, train_labels)

# -------------------------------
# ฟังก์ชันสำหรับวิเคราะห์อุบัติเหตุ
# -------------------------------
def analyze_accident(description):
    return model.predict([description])[0]

# -------------------------------
# Routes ของ Flask
# -------------------------------
@app.route("/")
def index():
    return render_template("page1.html")

@app.route("/page2", methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        data = request.get_json()
        description = data.get('description')
        if description:
            result = analyze_accident(description)
            return jsonify({"result": result})
        return jsonify({"error": "No description provided"}), 400
    return render_template("page2.html")

if __name__ == '__main__':  
   app.run(debug=True)
