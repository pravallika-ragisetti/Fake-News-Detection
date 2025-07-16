from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from src.preprocessing import load_and_clean_data

def train_tfidf_logistic():
    # Load cleaned dataset
    df = load_and_clean_data('data/fake_or_real_news.csv')

    # Drop rows with missing values just in case
    df = df.dropna(subset=['text', 'label'])

    # Ensure label is integer (0 or 1)
    df['label'] = df['label'].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Vectorization using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Save model and vectorizer
    joblib.dump(model, 'models/tfidf_logistic_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

    # Optional: Accuracy check
    score = model.score(X_test_vec, y_test)
    print(f"✅ TF-IDF Logistic Model Trained | Accuracy: {score:.4f}")
