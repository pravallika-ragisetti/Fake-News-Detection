import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text

def predict_with_logistic(news_text):
    model = joblib.load('models/tfidf_logistic_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "REAL" if pred == 1 else "FAKE"

def predict_with_lstm(news_text):
    model = load_model('models/lstm_model.h5')
    tokenizer = joblib.load('models/lstm_tokenizer.pkl')

    cleaned = clean_text(news_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=300)
    pred = model.predict(padded)[0][0]
    return "REAL" if pred >= 0.5 else "FAKE"
