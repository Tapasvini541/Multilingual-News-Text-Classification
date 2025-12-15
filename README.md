
# # 📰 Multi-class News Headline Analyzer

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
