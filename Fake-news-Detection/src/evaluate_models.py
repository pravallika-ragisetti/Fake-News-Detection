import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import load_and_clean_data
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression

def evaluate_tfidf_logistic():
    df = load_and_clean_data("data/fake_or_real_news.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/tfidf_logistic_model.pkl")

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print("📊 TF-IDF + Logistic Regression")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def evaluate_lstm():
    from src.preprocessing import clean_text
    tokenizer = joblib.load("models/lstm_tokenizer.pkl")
    model = load_model("models/lstm_model.h5")

    df = load_and_clean_data("data/fake_or_real_news.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    cleaned = X_test.apply(clean_text)
    sequences = tokenizer.texts_to_sequences(cleaned)
    X_test_pad = pad_sequences(sequences, maxlen=300)

    y_pred_probs = model.predict(X_test_pad)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    print("\n📊 LSTM Model")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
