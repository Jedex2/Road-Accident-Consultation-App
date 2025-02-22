# import flast module
from flask import Flask,render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
# instance of flask application
app = Flask(__name__)

# โหลดโมเดล BERT ภาษาไทย
model_name = "bert-base-thai"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# ฟังก์ชันในการประมวลผลข้อความ
def analyze_accident(description):
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()
# home route that returns below text when root url is accessed
@app.route("/")

def index():
    return render_template("page1.html")

@app.route("/page2", methods=['POST'])
def analyze():
    data = request.get_json()
    description = data.get('description')
    if description:
        result = analyze_accident(description)
        return jsonify({"result": result})
    return jsonify({"error": "No description provided"}), 400

@app.route('/page3')
def owner():
    return render_template("page3.html")

if __name__ == '__main__':  
   app.run()  
