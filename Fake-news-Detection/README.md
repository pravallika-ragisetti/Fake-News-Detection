# 📰 Fake News Detection using NLP 🧠

This project detects whether a news article is **real** or **fake** using Natural Language Processing (NLP) techniques with two models:
- ✅ TF-IDF + Logistic Regression  
- ✅ LSTM (Deep Learning)

---

## 🚀 Problem Statement

With the rapid growth of online news, fake news spreads faster than ever. This project aims to build a machine learning model to automatically **classify news articles as real or fake**, helping reduce misinformation.

---

## ✅ Solution

- Clean and preprocess the news text  
- Convert text to numerical form (TF-IDF or tokenized sequences)  
- Train and evaluate two models:  
  - **Logistic Regression** with TF-IDF  
  - **LSTM** using token sequences  
- Predict news authenticity based on model outputs

---

## ⚙️ Tech Stack (Short)

- **Language**: Python 3.10+  
- **Libraries**:  
  - 🧹 NLP: `NLTK`, `re`  
  - 📊 ML: `Scikit-learn`, `TF-IDF`, `LogisticRegression`  
  - 🤖 DL: `TensorFlow/Keras` (LSTM)  
  - 📁 Data: `Pandas`, `NumPy`, `joblib`  

---

## 🔁 Workflow

1. Load and merge `True.csv` & `Fake.csv`
2. Preprocess text (cleaning, stopword removal)
3. Train Logistic Regression and LSTM models
4. Evaluate performance (Accuracy, Precision, Recall)
5. Input custom news for real-time prediction

---

## 🌍 Real-World Usage (Short)

- **Social Media**: Flag suspicious content  
- **News Platforms**: Filter unreliable articles  
- **Browser Extensions**: Detect fakes in real time  
- **Govt. Monitoring**: Counter fake campaigns  
- **Educational Tools**: Teach media literacy  

---

## ✅ Feasibility (Short)

- Lightweight models for fast inference  
- Scalable with cloud or web APIs  
- Uses public datasets & free tools  
- Needs regular retraining for evolving language  
- Hard to detect satire or sarcasm accurately  

---

## 📦 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main app
python app.py
