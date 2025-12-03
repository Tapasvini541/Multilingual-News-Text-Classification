import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import gradio as gr
from langdetect import detect
from langcodes import Language
from deep_translator import GoogleTranslator, PonsTranslator
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import shap
from wordcloud import WordCloud
from newsapi import NewsApiClient
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import requests
from datetime import datetime
from newspaper import Article
import re

# ------------------- NLTK Resources -------------------
nltk.download('punkt')
nltk.download('stopwords')

# ------------------- Model Setup -------------------
device = torch.device("cpu")
print("Using device:", device)

model_name = "models/xlmr-agnews"  # your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

labels = ["🌍 World", "🏅 Sports", "💼 Business", "🔬 Sci/Tech"]

# ------------------- SHAP Helper -------------------
def hf_model_wrapper(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = list(texts)

    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

# ------------------- Summarization and Emotion Detection -------------------
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
except:
    summarizer = None

emotion_analyzer = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    return_all_scores=True,
    device=-1
)

rake = Rake(stopwords=stopwords.words("english"))

# ------------------- Helper Functions -------------------
def safe_translate(text, src_lang):
    if src_lang.lower() == "en":
        return text
    try:
        translation = GoogleTranslator(source='auto', target='en').translate(text)
        if translation.strip().lower() != text.strip().lower():
            return translation
    except:
        pass
    try:
        translation = PonsTranslator(source='auto', target='en').translate(text)
        if translation.strip().lower() != text.strip().lower():
            return translation
    except:
        pass
    return "(Translation may not be available)"

# ------------------- Emotion Detection -------------------
emotion_map = {
    "admiration": "Joy/Pride", "amusement": "Joy", "anger": "Anger", "annoyance": "Anger",
    "approval": "Joy/Pride", "caring": "Caring", "confusion": "Confusion", "curiosity": "Curiosity",
    "desire": "Desire", "disappointment": "Sadness", "disapproval": "Sadness",
    "embarrassment": "Embarrassment", "excitement": "Excitement", "fear": "Fear",
    "gratitude": "Gratitude", "grief": "Sadness", "joy": "Joy", "love": "Love",
    "nervousness": "Nervousness", "optimism": "Optimism", "pride": "Pride", "realization": "Realization",
    "relief": "Relief", "remorse": "Remorse", "sadness": "Sadness", "surprise": "Surprise",
    "neutral": "Neutral"
}

def detect_emotion(text, min_score=0.2, top_k=3):
    try:
        raw_emotions = emotion_analyzer(text)[0]
        filtered = [e for e in raw_emotions if e['score'] >= min_score]
        if not filtered:
            return "Neutral (no strong emotion)"
        mapped = {}
        for e in filtered:
            simple_label = emotion_map.get(e['label'], e['label'])
            if simple_label not in mapped or e['score'] > mapped[simple_label]:
                mapped[simple_label] = e['score']
        top_emotions = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ", ".join([f"{k} ({v:.2f})" for k,v in top_emotions])
    except:
        return "Emotion not available"

# ------------------- Single Headline Analysis -------------------
def analyze_headline(text):
    if not text.strip():
        fig, _ = plt.subplots(figsize=(5,3))
        return "⚠️ Please enter a headline", "", "", "", "", "", fig

    try:
        lang_code = detect(text)
        lang_name = Language.get(lang_code).display_name()
    except:
        lang_code = "unknown"
        lang_name = "Unknown"

    translation = safe_translate(text, lang_code)

    if summarizer:
        try:
            summary = summarizer(translation, max_length=32, min_length=10, do_sample=False)[0]['summary_text']
        except:
            summary = "(Summary not available)"
    else:
        summary = "(Summary not available)"

    rake.extract_keywords_from_text(translation)
    keywords = ', '.join(rake.get_ranked_phrases()[:5])

    emotion_text = detect_emotion(translation)

    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()[0]

    top_idx = np.argmax(probs)
    pred_text = f"{labels[top_idx]} ({probs[top_idx]*100:.2f}%)"

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(labels, probs*100, color=["#1E88E5","#43A047","#FB8C00","#8E24AA"])
    ax.set_ylim([0,100])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence")

    return pred_text, lang_name, translation, summary, keywords, emotion_text, fig

def single_headline_analysis(text):
    pred_text, lang, translation, summary, keywords, emotion_text, confidence_plot = analyze_headline(text)
    shap_plot = shap_explain_single(text)
    return pred_text, lang, translation, summary, keywords, emotion_text, confidence_plot, shap_plot

# ------------------- Single-headline SHAP -------------------
def shap_explain_single(text, labels=labels):
    """Generate SHAP explainability plot for a single headline."""
    explainer = shap.Explainer(hf_model_wrapper, masker=shap.maskers.Text(tokenizer))
    shap_values = explainer([text])[0]  # single headline

    # Find class index with max model output probability
    probs = hf_model_wrapper([text])[0]
    class_idx = int(np.argmax(probs))
    label_name = labels[class_idx]

    # Get top contributing words
    top_words = []
    if shap_values.values.ndim == 2:  # shape (tokens, classes)
        # Get the SHAP values for the predicted class
        class_shap = shap_values.values[:, class_idx]
    else:  # fallback
        class_shap = shap_values.values

    # Take top 10 words
    token_names = shap_values.data if hasattr(shap_values, 'data') else [f"Word{i}" for i in range(len(class_shap))]
    sorted_idx = np.argsort(np.abs(class_shap))[-10:][::-1]
    top_words = [token_names[i] for i in sorted_idx]
    impacts = class_shap[sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(top_words, impacts, color="#1E88E5")
    ax.set_xlabel("Impact on Prediction")
    ax.set_title(f"SHAP Explanation for '{label_name}'")
    plt.tight_layout()
    return fig


# ------------------- Batch Analytics -------------------
def cluster_headlines(headlines, n_clusters=5):
    model_embed = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    embeddings = model_embed.encode(headlines)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings_scaled)

    cluster_summary = []
    for cid in range(n_clusters):
        cluster_texts = [headlines[i] for i in range(len(headlines)) if cluster_ids[i]==cid]
        all_words = ' '.join(cluster_texts).split()
        top_words = pd.Series(all_words).value_counts().head(5).index.tolist()
        cluster_summary.append({'Cluster': cid, 'Num Headlines': len(cluster_texts), 'Top Keywords': ', '.join(top_words)})
    return cluster_ids, pd.DataFrame(cluster_summary)

def detect_anomalies(pred_probs, threshold=0.6):
    max_probs = np.max(pred_probs, axis=1)
    anomalies = np.where(max_probs < threshold)[0]
    return anomalies

def shap_explain_batch(headlines):
    explainer = shap.Explainer(hf_model_wrapper, masker=shap.maskers.Text(tokenizer))
    shap_values = explainer(headlines)

    top_words = {label: [] for label in labels}
    for sv in shap_values:
        # Skip if dimensions don't match
        if sv.values.ndim != 2 or sv.values.shape[1] != len(labels):
            continue

        class_idx = int(np.argmax(sv.values))
        class_idx = min(class_idx, len(labels)-1)  # safe indexing
        label_name = labels[class_idx]

        if hasattr(sv, 'data'):
            sorted_idx = np.argsort(np.abs(sv.values[:, class_idx]))[-5:]
            top_words[label_name].extend([sv.data[i] for i in sorted_idx])

    fig, ax = plt.subplots(figsize=(6,4))
    counts = [len(top_words[label]) for label in labels]
    ax.bar(labels, counts, color="#1E88E5")
    ax.set_ylabel("Top contributing words count")
    ax.set_title("SHAP Explainability per Class")
    return fig



# ------------------- Interactive Visuals -------------------
def wordcloud_all_headlines(df):
    # Combine all keywords from all headlines
    text = ' '.join(df['Keywords'].dropna())
    if not text:
        text = "No keywords available"
    wc = WordCloud(width=600, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud for All Headlines")
    return fig

def emotion_distribution(df):
    all_emotions = []
    for emo in df['Emotions']:
        for part in emo.split(','):
            label = part.split('(')[0].strip()
            all_emotions.append(label)
    counts = pd.Series(all_emotions).value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title("Emotion Distribution Across Headlines")
    return fig

def topic_prevalence(df):
    topics = [pred.split('(')[0].strip() for pred in df['Prediction']]
    counts = pd.Series(topics).value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index, counts.values, color="#1E88E5")
    ax.set_ylabel("Number of Headlines")
    ax.set_title("Topic Prevalence")
    return fig

def export_processed_df(df):
    filename = "processed_headlines.csv"
    df.to_csv(filename, index=False)
    return filename

def batch_analytics(file):
    df = pd.read_csv(file.name)
    headlines = df.iloc[:,0].astype(str).tolist()
    all_preds, all_probs, all_emotions, all_keywords = [], [], [], []
    all_langs, all_translations = [], []

    for h in headlines:
        pred_text, lang, translation, summary, keywords, emotions, fig = analyze_headline(h)
        all_preds.append(pred_text)
        all_langs.append(lang)
        all_translations.append(translation)
        prob_values = [float(x.split('(')[1].replace('%','').replace(')',''))/100 for x in pred_text.split(',') if '(' in x]
        all_probs.append(prob_values if prob_values else [0.0]*len(labels))
        all_emotions.append(emotions)
        all_keywords.append(keywords)

    cluster_ids, cluster_table = cluster_headlines(headlines)
    anomalies_idx = detect_anomalies(np.array(all_probs))
    anomaly_table = pd.DataFrame({
        'Headline':[headlines[i] for i in anomalies_idx],
        'Prediction':[all_preds[i] for i in anomalies_idx]
    })

    shap_fig = shap_explain_batch(headlines)
    processed_df = pd.DataFrame({
        "Headline": headlines,
        "Detected Language": all_langs,
        "English Translation": all_translations,
        "Prediction": all_preds,
        "Keywords": all_keywords,
        "Emotions": all_emotions,
        "Cluster": cluster_ids
    })
    csv_file = export_processed_df(processed_df)
    wc_fig = wordcloud_all_headlines(processed_df)
    emo_fig = emotion_distribution(processed_df)
    topic_fig = topic_prevalence(processed_df)
    return processed_df, cluster_table, anomaly_table, shap_fig, wc_fig, emo_fig, topic_fig, csv_file

newsapi = NewsApiClient(api_key="4cb177dc4906430391984668f34f3ccd")  # replace with your key

def fetch_latest_headlines(page_size=3):
    """
    Fetch 3 news articles per language across:
    English, French, German, Spanish, Hindi, Tamil, Italian.
    """

    API_KEY = "323e4b80dec34889a5750e56087a2140"
    BASE_URL = "https://newsapi.org/v2/everything"
    LANGUAGES = ["en", "fr", "de", "es", "hi", "ta", "it","ar","pt","ru","zh"]

    all_articles = []

    for lang in LANGUAGES:
        params = {
            "apiKey": API_KEY,
            "language": lang,
            "q": "news",              # required for /everything
            "sortBy": "publishedAt",
            "pageSize": page_size
        }

        lang_name = Language.get(lang).display_name()
        print(f"🌐 Fetching {page_size} {lang_name} news articles...")

        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"❌ Error fetching {lang_name}: {e}")
            continue

        if data.get("status") != "ok" or not data.get("articles"):
            print(f"⚠️ No news returned for {lang_name} ({lang})")
            continue

        for article in data["articles"]:
            all_articles.append({
                "Headline": article.get("title", "(No title)"),
                "Description": article.get("description", "(No description)"),
                "PublishedAt": article.get("publishedAt", ""),
                "Language": lang_name,
                "Source": article.get("source", {}).get("name", ""),
                "URL": article.get("url", "")
            })

    df = pd.DataFrame(all_articles)
    print(f"\n✅ Total fetched: {len(df)} articles\n")
    return df


# -------------------------------
# 2️⃣ TRANSLATION HELPERS
# -------------------------------
def robust_translate(text, target="en"):
    text = str(text).strip()
    if not text:
        return "(No text available)"
    try:
        detected = detect(text)
    except:
        detected = "en"

    if detected.lower() == target.lower():
        return text

    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except:
        try:
            return PonsTranslator(source='auto', target=target).translate(text)
        except:
            return text


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


# -------------------------------
# 3️⃣ PROCESS & CLASSIFY HEADLINES
# -------------------------------
def process_live_headlines():
    """
    Fetch, translate, classify, and analyze live multilingual news headlines
    using the XLM-R model and GoEmotions pipeline.
    """
    df = fetch_latest_headlines()
    df = df.drop_duplicates(subset=["Headline"], keep="first")

    results = []

    for _, row in df.iterrows():
        headline = row["Headline"]
        desc = row["Description"]
        lang = row["Language"]

        # 🔹 Translate headline & description
        headline_en = robust_translate(headline)
        desc_en = robust_translate(desc)
        headline_en = clean_text(headline_en)
        desc_en = clean_text(desc_en)

        # 🔹 Use transformer model for classification
        enc = tokenizer(headline_en, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        prediction = f"{labels[top_idx]} ({probs[top_idx]*100:.2f}%)"

        # 🔹 Extract keywords
        rake.extract_keywords_from_text(headline_en + " " + desc_en)
        keywords = ', '.join(rake.get_ranked_phrases()[:5])

        # 🔹 Emotion detection
        emotion_text = detect_emotion(headline_en)

        # 🔹 Add to results
        results.append({
            "Headline": headline,
            "Translated Headline": headline_en,
            "Description": desc_en,
            "PublishedAt": row["PublishedAt"],
            "Detected Language": lang,
            "Prediction": prediction,
            "Keywords": keywords,
            "Emotions": emotion_text
        })

    return pd.DataFrame(results)

# -------------------------------
# 4️⃣ EXPORT TO PDF
# -------------------------------
def export_news_pdf(df, filename="classified_news.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                 fontSize=22, textColor=colors.HexColor("#1E88E5"), spaceAfter=12)
    headline_style = ParagraphStyle('Headline', parent=styles['Heading2'],
                                    fontSize=14, textColor=colors.black, spaceAfter=4)
    desc_style = ParagraphStyle('Desc', parent=styles['Normal'],
                                fontSize=11, leading=15, spaceAfter=10)
    emotion_style = ParagraphStyle('Emotion', parent=styles['Italic'],
                                   textColor=colors.grey)

    story = []
    story.append(Paragraph("📰 Classified News Report", title_style))
    story.append(Spacer(1, 12))

    # Group by Prediction
    for label in df["Prediction"].unique():
        section_df = df[df["Prediction"] == label]
        if not section_df.empty:
            story.append(PageBreak())
            story.append(Paragraph(f"📂 {label}", styles['Heading1']))
            story.append(Spacer(1, 8))

            for _, row in section_df.iterrows():
                story.append(Paragraph(f"<b>{row['Translated Headline']}</b>", headline_style))
                story.append(Paragraph(f"<i>{row['PublishedAt']}</i>", styles['Normal']))
                story.append(Paragraph(row['Description'], desc_style))
                story.append(Paragraph(f"💬 Emotions: {row['Emotions']}", emotion_style))
                story.append(Spacer(1, 12))

    doc.build(story)
    return filename

# ------------------- Gradio UI -------------------
css = """
#title {text-align:center; font-size:32px; font-weight:bold; color:#1E88E5; margin-bottom:5px;}
#desc {text-align:center; font-size:16px; color:#555; margin-bottom:20px;}
.gr-button {background-color:#1E88E5; color:white; font-weight:bold; border-radius:10px; padding:8px;}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<div id='title'>🌐 Multi-class News Headline Analyzer for different Languages</div>")
    gr.HTML("<div id='desc'>Single & Batch headline analysis with SHAP explainability, clustering, keyword-emotion tracking, and interactive analytics.</div>")

    with gr.Tabs():
        with gr.TabItem("Single Headline"):
            with gr.Row():
                with gr.Column(scale=3):
                    input_box = gr.Textbox(lines=3, placeholder="✍️ Enter a news headline...", label="Your Headline")
                    classify_btn = gr.Button("🚀 Analyze", variant="primary")
                with gr.Column(scale=5):
                    output_label = gr.Textbox(label="Prediction", interactive=False, lines=2)
                    lang_box = gr.Textbox(label="Detected Language", interactive=False, lines=1)
                    translation_box = gr.Textbox(label="English Translation", interactive=False, lines=4)
                    summary_box = gr.Textbox(label="Summary", interactive=False, lines=4)
                    keywords_box = gr.Textbox(label="Keywords", interactive=False, lines=2)
                    emotion_box = gr.Textbox(label="Emotions", interactive=False, lines=2)
                    confidence_plot = gr.Plot(label="Confidence per Class")
                    shap_single_plot = gr.Plot(label="SHAP Explainability")
            classify_btn.click(
                single_headline_analysis,
                inputs=input_box,
                outputs=[output_label, lang_box, translation_box, summary_box, keywords_box, emotion_box, confidence_plot, shap_single_plot]
                )

        with gr.TabItem("Batch Upload"):
            with gr.Row():
                with gr.Column(scale=3):
                    batch_file = gr.File(label="📄 Upload CSV (first column = headlines)")
                    process_btn = gr.Button("⚡ Process Batch")
                    download_btn = gr.File(label="📥 Download Processed CSV")
                with gr.Column(scale=5):
                    processed_table = gr.Dataframe(label="Processed Headlines")
                    cluster_table_box = gr.Dataframe(label="Cluster Summary")
                    anomaly_table_box = gr.Dataframe(label="Anomalies / Outliers")
                    shap_plot = gr.Plot(label="SHAP Explainability (Batch)")
                    wordcloud_plot = gr.Plot(label="Word Cloud per Cluster")
                    emotion_dist_plot = gr.Plot(label="Emotion Distribution")
                    topic_plot = gr.Plot(label="Topic Prevalence")

            process_btn.click(
                batch_analytics,
                inputs=batch_file,
                outputs=[processed_table, cluster_table_box, anomaly_table_box, shap_plot, wordcloud_plot, emotion_dist_plot, topic_plot, download_btn])
        
        with gr.TabItem("Real-time News Feed"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_box = gr.Textbox(label="(Optional) Search Query", placeholder="Leave empty to fetch all news")
                    refresh_btn = gr.Button("🔄 Fetch Latest News")
                    pdf_btn = gr.Button("📄 Generate PDF Report")
                with gr.Column(scale=5):
                    live_table = gr.Dataframe(label="Live News Feed", interactive=False)
                    pdf_file = gr.File(label="📥 Download News Report")

            refresh_btn.click(process_live_headlines, None, live_table)

            def generate_news_pdf():
                df = process_live_headlines()
                file_path = export_news_pdf(df)
                return file_path

            pdf_btn.click(generate_news_pdf, None, pdf_file)

demo.launch()
