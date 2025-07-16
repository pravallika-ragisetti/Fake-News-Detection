import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from src.preprocessing import load_and_clean_data

def train_lstm():
    # Load cleaned dataset
    df = load_and_clean_data('data/fake_or_real_news.csv')

    # Tokenization
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['text'])

    sequences = tokenizer.texts_to_sequences(df['text'])

    # Padding
    max_len = 300
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['label'].values

    # Save tokenizer for future predictions
    import joblib
    joblib.dump(tokenizer, 'models/lstm_tokenizer.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model architecture
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # Save model
    model.save('models/lstm_model.h5')
    print("✅ LSTM model trained and saved.")
