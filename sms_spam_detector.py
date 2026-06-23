"""
SMS Spam Detector - Natural Language Processing (NLP) Basics
This script demonstrates how to build a Machine Learning model to classify SMS messages as 'spam' or 'ham' (not spam).
It introduces core NLP concepts: Tokenization, Stop Word Removal, and TF-IDF Vectorization.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("🚀 Initializing SMS Spam Detector Pipeline...\n")

    # ==========================================
    # STEP 1: Load and Inspect the Data
    # ==========================================
    # The dataset is a TSV (Tab-Separated Values) file without headers.
    # We assign columns 'label' (ham/spam) and 'message' (the text).
    try:
        df = pd.read_csv('dataset/SMSSpamCollection', sep='\t', names=['label', 'message'])
        print(f"✅ Dataset loaded successfully. Total messages: {len(df)}")
    except FileNotFoundError:
        print("❌ Dataset not found. Please ensure 'SMSSpamCollection' is in the 'dataset/' directory.")
        return

    # Convert the text labels ('ham', 'spam') into binary numbers (0, 1) for the ML model.
    # Neural Networks and ML algorithms understand math, not text!
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # ==========================================
    # STEP 2: Train/Test Split
    # ==========================================
    # We split the data into 80% for training and 20% for testing.
    # X contains the raw text messages, y contains the labels (0 or 1).
    X = df['message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Training on {len(X_train)} messages, Testing on {len(X_test)} messages.\n")

    # ==========================================
    # STEP 3: Natural Language Processing (NLP)
    # Feature Extraction using TF-IDF
    # ==========================================
    # Machine Learning models cannot process raw text (like "Win a free iPhone!").
    # We must convert text into numbers (vectors). We use TF-IDF (Term Frequency-Inverse Document Frequency).
    # 
    # What TF-IDF does:
    # 1. Tokenization: Splits sentences into individual words.
    # 2. Lowercasing: Converts all text to lowercase ("Win" -> "win").
    # 3. Stop Words Removal: Removes common words ("the", "is", "in") that have no predictive power.
    # 4. Scoring: Gives a high score to a word if it appears frequently in a single message (TF),
    #    but heavily penalizes the word if it appears in almost every message across the dataset (IDF).
    #    (e.g., the word "FREE" is rare overall, but appears heavily in spam messages, so it gets a high score).
    
    print("🧠 Converting Text to Math using TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the training data (learn the vocabulary) and transform it into numbers.
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Only transform the test data (do NOT fit, otherwise the model cheats by seeing test vocabulary!)
    X_test_tfidf = vectorizer.transform(X_test)

    # ==========================================
    # STEP 4: Train the ML Model (Naive Bayes)
    # ==========================================
    # Naive Bayes is an algorithm based on Bayes' Theorem of probability. 
    # It is historically the most famous and effective algorithm for text classification (spam filtering).
    # It calculates the probability that a message is spam given the specific words it contains.
    
    print("🤖 Training the Naive Bayes Classifier...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # ==========================================
    # STEP 5: Evaluation
    # ==========================================
    print("⚖️ Evaluating Model Performance...\n")
    predictions = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, predictions)
    print(f"🏆 Overall Accuracy: {acc * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    # target_names maps the binary labels back to human-readable strings for the report
    print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
    
    # ==========================================
    # STEP 6: Interactive Testing (Inference)
    # ==========================================
    print("\n🔮 Testing custom messages:")
    custom_messages = [
        "Hey man, are we still going to the game tonight?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        "Can you pick up milk on your way home?"
    ]
    
    # To predict new messages, we must process them through the EXACT SAME vectorizer we trained earlier.
    custom_tfidf = vectorizer.transform(custom_messages)
    custom_preds = model.predict(custom_tfidf)
    
    for msg, pred in zip(custom_messages, custom_preds):
        label = "🚨 SPAM" if pred == 1 else "✅ HAM"
        print(f"[{label}] -> {msg}")

if __name__ == "__main__":
    main()
