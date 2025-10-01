from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS so other apps can call this API

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        news_text = data.get("content", "")

        if not news_text:
            return jsonify({"error": "No content provided"}), 400

        # Transform input text
        text_tfidf = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        label = "FAKE" if prediction == 1 else "REAL"

        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Local run (Render will ignore this and use gunicorn)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
