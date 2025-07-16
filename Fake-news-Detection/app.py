from src.merge_datasets import merge_true_fake
from src.tfidf_logistic import train_tfidf_logistic
from src.lstm_model import train_lstm
from src.predict import predict_with_logistic, predict_with_lstm
from src.evaluate_models import evaluate_tfidf_logistic, evaluate_lstm

def main():
    print("🔄 Merging True and Fake news datasets...")
    merge_true_fake()

    print("\n🧠 Training TF-IDF + Logistic Regression model...")
    train_tfidf_logistic()

    print("\n🧠 Training LSTM model...")
    train_lstm()

    print("\n📝 You can now enter a news article below to check if it's REAL or FAKE.")
    print("Type 'exit' to quit the input loop.\n")

    while True:
        try:
            sample_news = input("📰 News Text > ")
            if sample_news.strip().lower() == "exit":
                print("👋 Exiting prediction loop.")
                break

            if not sample_news.strip():
                print("⚠️ Please enter valid news text.")
                continue

            tfidf_result = predict_with_logistic(sample_news)
            lstm_result = predict_with_lstm(sample_news)

            print(f"\n📌 Prediction Results:")
            print(f"🔎 TF-IDF Logistic: {tfidf_result}")
            print(f"🔎 LSTM Model:      {lstm_result}")
            print("-" * 50)

        except Exception as e:
            print("❌ Error during prediction:", e)
            break

    print("\n📈 Evaluating models on test data...")
    evaluate_tfidf_logistic()
    evaluate_lstm()

if __name__ == "__main__":
    main()
