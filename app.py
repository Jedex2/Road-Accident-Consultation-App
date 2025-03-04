from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

app = Flask(__name__)

# -------------------------------
# Training Data for Bayesian Analysis
# -------------------------------
train_texts = [
    "I was hit from behind while stopping at a red light.",  
    "I hit a pedestrian crossing the road.",
    "My car was scratched by another vehicle.",
    "I had the right of way.",
    "I might be at fault because I changed lanes suddenly.",
    "I hit a motorcyclist who ran a red light.",
    "A car exited an alley and hit me from the side."
]

train_labels = [
    "Other driver at fault",  # Case 1: Rear-end collision, usually other driver at fault
    "You might be at fault",  # Case 2: Hitting a pedestrian is usually the driver's fault
    "Minor accident",  # Case 3: Small scratch, minor case
    "You are correct",  # Case 4: The driver had the right of way
    "You might lose the case",  # Case 5: Sudden lane change is risky
    "Other driver at fault",  # Case 6: Motorcyclist ran red light
    "Other driver at fault"  # Case 7: Exiting from alley without yielding
]

# Create and train model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
model = MultinomialNB()
model.fit(X_train, train_labels)

# -------------------------------
# Function for Accident Analysis
# -------------------------------
def analyze_accident(description):
    X_input = vectorizer.transform([description])
    return model.predict(X_input)[0]

# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("page1.html")

@app.route("/page2", methods=["GET", "POST"])  # ✅ แก้ตรงนี้
def page2():
    if request.method == "POST":
        data = request.get_json()
        description = data.get("description")
        if description:
            result = analyze_accident(description)
            return jsonify({"result": result})
        return jsonify({"error": "No description provided"}), 400
    return render_template("page2.html")


if __name__ == '__main__':  
   app.run(debug=True)