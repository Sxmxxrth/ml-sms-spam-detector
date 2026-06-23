# 📱 NLP SMS Spam Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green.svg)

A Natural Language Processing (NLP) pipeline built to classify SMS messages as either Spam or Ham (Not Spam). This project demonstrates the foundational concepts of converting human language into mathematical vectors for Machine Learning classification.

## 🧠 Technical Architecture

This project utilizes a classic NLP pipeline:
1. **Data Preprocessing**: Cleaning text and formatting labels.
2. **Tokenization & Stop Words**: Removing common, low-value English words ("the", "is", "a").
3. **TF-IDF Vectorization**: Converting words into numerical matrices using Term Frequency-Inverse Document Frequency to heavily score rare, predictive words.
4. **Multinomial Naive Bayes**: Training a highly efficient probabilistic classifier based on Bayes' Theorem to predict spam probability based on word occurrences.

## 📊 Dataset
The model is trained on the classic [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Sxmxxrth/ml-sms-spam-detector.git
cd ml-sms-spam-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the pipeline**
```bash
python sms_spam_detector.py
```

## 📈 Results
The model achieves over **96% accuracy** with extremely high precision on Spam detection, ensuring that valid messages are not accidentally sent to the spam folder.

## 📝 License
MIT License
