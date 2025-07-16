import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords once (if not already)
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', str(text))

    # Lowercase and tokenize
    words = text.lower().split()

    # Remove stopwords and apply stemming
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]

    return ' '.join(cleaned_words)

def load_and_clean_data(filepath):
    # Load the CSV file
    df = pd.read_csv(filepath)

    # Drop rows with missing text or label
    df = df.dropna(subset=['text', 'label'])

    # Apply text cleaning
    df['text'] = df['text'].apply(clean_text)

    # Convert labels to integers (in case they are floats or strings)
    df['label'] = df['label'].astype(int)

    return df
