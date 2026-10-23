# FAQ Chatbot

A simple NLP-based FAQ chatbot built with Python, Flask, NLTK and Scikit-learn.

## Project requirements covered

1. Collect FAQs in `faq_data.json`.
2. Preprocess text using NLTK stemming and stopword removal.
3. Convert FAQ questions into TF-IDF vectors.
4. Match the user's question using cosine similarity.
5. Display the best matching answer.
6. Provide a simple web chat UI.

## How to run

### 1. Open the project folder
```bash
cd FAQ_Chatbot_Project
```

### 2. Create a virtual environment (optional)
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### 3. Install libraries
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python app.py
```

### 5. Open in browser
Go to:
`http://127.0.0.1:5000`

## How to customize

Open `faq_data.json` and add your own question-answer pairs. For example:

```json
{
  "question": "How can I reset my password?",
  "answer": "Click on Forgot Password and follow the instructions."
}
```

Restart the application after changing the FAQ file.

## Algorithm

User Question
      ↓
Text Preprocessing (NLTK)
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Best FAQ Match
      ↓
Answer + Confidence
      ↓
Chatbot UI
