from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model and TfidfVectorizer
loaded_model, loaded_tfidf = joblib.load('car_accident_model.pkl')

# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("page1.html")

@app.route("/page2", methods=["GET", "POST"])
def page2():
    if request.method == "POST":
        data = request.get_json()
        description = data.get("description")
        if description:
            prediction = loaded_model.predict(loaded_tfidf.transform([description]))
            result = 'Plaintiff Wins' if prediction[0] == 1 else 'Defendant Wins'
            return jsonify({"result": result})
        return jsonify({"error": "No description provided"}), 400
    return render_template("page2.html")

if __name__ == '__main__':
    app.run(debug=True)