
## 📰 Multi-class News Headline Analyzer

An interactive **Machine Learning and NLP-based web application** that analyzes news headlines across multiple languages to classify them into **categories**, detect **emotions**, extract **keywords**, and provide **SHAP explainability**, built using **Gradio**, **transformer-based models**, and classical **NLP techniques**.


## 🚀 Project Overview

This project fetches live news headlines using the NewsAPI, preprocesses multilingual text, and applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to:

- Classify headlines into **World, Sports, Business, Sci/Tech**
- Detect **emotions** in headlines
- Extract **keywords and summaries**
- Provide **SHAP explainability** for model predictions
- Visualize **emotion distribution, topic prevalence, and keyword cloud**

The application provides a **user-friendly web interface** for real-time analysis.

## 📂 Project Structure

```
multilingual-newstext-classifcation/
├── app.py                       
├── requirements.txt         
└── README.md                   
```
## ✨ Key Features

- 🌐 **Fetch latest news headlines from multiple languages**  
- 🔹 **Single headline analysis** with classification, summary, keywords, emotion detection, and SHAP explainability  
- ⚡ **Batch processing** of multiple headlines via CSV  
- 🖼️ **Keyword cloud visualization**  
- 📊 **Emotion distribution and topic prevalence plots**  
- 📰 **Real-time news feed monitoring**  
- 📄 **Export processed results** as CSV or PDF  
- 🖥️ **Interactive Gradio UI**
## 🛠️ Tech Stack

### **Programming & Frameworks**
- Python  
- Gradio  

### **Machine Learning & NLP**
- Hugging Face Transformers (XLM-R, BART, RoBERTa-GoEmotions)  
- PyTorch  
- NLTK  
- RAKE  
- Sentence-Transformers (for clustering)  

### **Visualization**
- Matplotlib  
- WordCloud  
- SHAP  

### **APIs**
- NewsAPI  
- Google Translator API  
- Pons Translator API

## 🧠 Machine Learning Workflow

- **Fetch latest headlines** using NewsAPI  
- **Detect language** & translate non-English headlines  
- **Clean text** and extract keywords using RAKE  
- **Summarize headlines** using BART Transformer  
- **Classify headlines** with fine-tuned XLM-R model  
- **Detect emotions** using RoBERTa GoEmotions  
- **Provide SHAP explainability**  
- **Perform clustering** using sentence embeddings + KMeans  
- **Visualize emotion distribution, topic prevalence, and keyword clouds**
  
## 🧪 Application Modules

### 1️⃣ Single Headline Analysis
- Enter a headline to get **category, confidence, keywords, summary, emotions, SHAP plot**

### 2️⃣ Batch Upload
- Upload **CSV** with headlines  
- Get **processed table, cluster summary, anomaly detection, word cloud, emotion distribution, and topic prevalence plots**

### 3️⃣ Real-time News Feed
- Fetch latest headlines across multiple languages  
- Optional **search query filtering**  
- Download **PDF report** of classified news

## 🖥️ How to Run the Project

### 🔧 Prerequisites
- Python 3.8 or higher
- YouTube Data API Key

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run the Application
```bash
streamlit run app.py
```
Or manually install:

```bash
pip install torch transformers gradio nltk pandas numpy matplotlib seaborn wordcloud sentence-transformers shap newspaper3k reportlab langdetect deep-translator requests
```





## 📌 Results & Insights

### Model Performance Summary

| Model | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.9164 | 0.9163 | 0.9164 | 0.9163 |
| Transformer (XLM-RoBERTa) | 0.9421 | 0.9402 | 0.9410 | 0.9405 |

## 🌱 Future Enhancements

- Fine-tuned **multilingual transformer models**  
- **Real-time alerts** for breaking news  
- **Topic modeling** and trend detection  
- **Cloud deployment** for scalable analytics

## 👩‍💻 Author

**Tapasvini S**  
🎓 MSc Artificial Intelligence & Machine Learning(5 years Integrated course)  
📧 Email: tapasvini541@gmail.com  
🔗 GitHub: https://github.com/Tapasvini541  
## ⭐ Acknowledgements
- Hugging Face **Transformers & Datasets**  
- **NewsAPI**  
- **NLTK** & **Sentence-Transformers Community**  
- **SHAP Library** for explainable AI
