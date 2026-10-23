from flask import Flask, render_template, request, jsonify
import json
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

stemmer = PorterStemmer()
STOPWORDS = {
    "a","an","the","is","are","am","was","were","be","been","being",
    "i","me","my","we","our","you","your","he","she","it","they","their",
    "what","which","who","whom","this","that","these","those",
    "do","does","did","can","could","will","would","should","how",
    "to","of","in","on","for","from","with","and","or","but","as","at",
    "about","into","by","please","tell","me"
}

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

questions = [item["question"] for item in faq_data]
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_matrix = vectorizer.fit_transform(processed_questions)

def find_answer(user_question):
    processed = preprocess(user_question)
    if not processed.strip():
        return "Please enter a question.", 0.0

    user_vector = vectorizer.transform([processed])
    scores = cosine_similarity(user_vector, faq_matrix)[0]
    best_index = scores.argmax()
    confidence = float(scores[best_index])

    if confidence < 0.18:
        return (
            "Sorry, I couldn't find a close match in my FAQ database. "
            "Please try asking in a different way or contact support.",
            confidence
        )

    return faq_data[best_index]["answer"], confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    answer, confidence = find_answer(question)
    return jsonify({
        "answer": answer,
        "confidence": round(confidence * 100, 1)
    })

if __name__ == "__main__":
    app.run(debug=True)
