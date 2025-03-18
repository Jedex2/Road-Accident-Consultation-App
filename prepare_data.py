import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
import nltk

# ดาวน์โหลด NLTK resources (ถ้าจำเป็น)
nltk.download('punkt')

# โหลดข้อมูลจาก CSV
df = pd.read_csv('Sheet7.csv')

# 1. ทำความสะอาดข้อมูล
# ลบข้อมูลที่ขาดหาย
df.dropna(inplace=True)

# 2. จัดรูปแบบข้อมูล
# แยกฟีเจอร์และเลเบล รวมเหตุผลด้วย
X = df[['Description', 'Reasoning', 'Relevant Laws']]
y = df['Winning Probability'].apply(lambda x: 1 if 'Plaintiff' in x else 0)

# 3. การเข้ารหัสข้อมูล
# ใช้ TF-IDF สำหรับคำอธิบายและเหตุผล
tfidf_vectorizer = TfidfVectorizer(max_features=500)

# ใช้ One-Hot Encoding สำหรับกฎหมายที่เกี่ยวข้อง
encoder = OneHotEncoder(handle_unknown='ignore')

# 4. แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. สร้าง Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', tfidf_vectorizer, 'Description'),
        ('reasoning', tfidf_vectorizer, 'Reasoning'),  # เพิ่มเหตุผลในการเข้ารหัส
        ('law', encoder, ['Relevant Laws'])  # Ensure 'Relevant Laws' is treated as a DataFrame
    ]
)

# สร้าง Pipeline ที่รวมการทำ Preprocessing และโมเดล
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())  # เปลี่ยนเป็น LogisticRegression
])

# 6. ฝึกสอนโมเดล
pipeline.fit(X_train, y_train)

# 7. ทดสอบโมเดล
accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')