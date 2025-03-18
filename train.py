import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE  # เพิ่ม SMOTE
import joblib
from collections import Counter


# **Step 1: Load Data**
# อ่านข้อมูลจากไฟล์ sheet7.csv
df = pd.read_csv("sheet7.csv")

# ตรวจสอบข้อมูลตัวอย่าง
print("Data Sample:")
print(df.head())  # แสดงตัวอย่างข้อมูล

# **Step 2: Clean Data**
# ฟังก์ชันแปลง Winning Probability
def extract_label_and_probability(prob_text):
    """
    แปลงข้อความ Winning Probability เป็น:
    - Label: 1 (สำหรับ Plaintiff ชนะ), 0 (สำหรับ Defendant ชนะ)
    - Probability: ค่าความน่าจะเป็นในรูปแบบตัวเลข (0.6 - 0.9)
    """
    match = re.search(r"(Plaintiff|Defendant) Win: (\d+)%", str(prob_text))
    if match:
        winner = match.group(1)  # Plaintiff หรือ Defendant
        probability = float(match.group(2)) / 100  # แปลงเปอร์เซ็นต์เป็นทศนิยม
        label = 1 if winner == "Plaintiff" else 0  # Plaintiff = 1, Defendant = 0
        return label, probability
    return None, 0.0  # ในกรณีที่ไม่มีข้อมูล

# ใช้ฟังก์ชันเพื่อแยก Label และ Probability
df[['Label', 'Winning Probability']] = df['Winning Probability'].apply(
    lambda x: pd.Series(extract_label_and_probability(x))
)
# ลบแถวที่มีค่า NaN ในคอลัมน์ 'Label'
df = df.dropna(subset=['Label'])
# แสดงข้อมูลที่ทำความสะอาดแล้ว
print("Cleaned Data:")
print(df[['Description', 'Winning Probability', 'Label']])

# **Step 3: Train-Test Split**
X = df['Description']  # Features (ข้อความ)
y = df['Label']        # Labels (Binary: 1 (Plaintiff ชนะ) หรือ 0 (Defendant ชนะ))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Step 3.1: Apply SMOTE**
# ใช้ TfidfVectorizer เพื่อแปลงข้อความเป็นเวกเตอร์ก่อนทำ SMOTE
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

# ใช้ SMOTE เพื่อปรับสมดุลข้อมูล
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# ตรวจสอบจำนวนตัวอย่างในแต่ละคลาสหลังจากทำ SMOTE
from collections import Counter
print("Balanced Data (SMOTE):", Counter(y_train_resampled))

# **Step 4: Create and Train Model**
# โมเดล Naive Bayes
model = MultinomialNB()

# เทรนโมเดลด้วยข้อมูลที่สมดุล
model.fit(X_train_resampled, y_train_resampled)

# **Step 5: Evaluate Model**
# แปลง X_test ด้วย TfidfVectorizer ก่อนทำนาย
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ดูจำนวนแต่ละ Label
print("Label Distribution in Original Data:")
print(df['Label'].value_counts())
# ตรวจสอบจำนวนตัวอย่างในแต่ละคลาสหลังการปรับสมดุลด้วย SMOTE
print("Balanced Data (SMOTE):", Counter(y_train_resampled))
# **Step 6: Save Model**
# บันทึกทั้งโมเดลและ TfidfVectorizer เพื่อใช้งานในอนาคต
# joblib.dump((model, tfidf), 'car_accident_model.pkl')
# print("Model saved as 'car_accident_model.pkl'")

# **Step 7: Load Model (ตัวอย่างการใช้งาน)**
loaded_model, loaded_tfidf = joblib.load('car_accident_model.pkl')
prediction = loaded_model.predict(loaded_tfidf.transform(["A vehicle was involved in an accident while trying to navigate through a wedding procession blocking the street."]))
print(f"Prediction: {'Plaintiff Wins' if prediction[0] == 1 else 'Defendant Wins'}")