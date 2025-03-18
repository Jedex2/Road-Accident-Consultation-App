import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# โหลดโมเดล
loaded_model = joblib.load('car_accident_model.pkl')

# ทำนายข้อความใหม่
new_description = ["A vehicle collided with a wedding procession that was blocking the street."]
prediction = loaded_model.predict(new_description)

# แสดงผลการทำนาย
print(f"Prediction: {'Plaintiff Wins' if prediction[0] == 1 else 'Defendant Wins'}")