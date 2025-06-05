import os
import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from better_profanity import profanity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Siftly", page_icon="🎭", layout="wide")

st.title("🎭 Siftly")
MODEL_PATH = "models/distilbert‑review‑lora‑merged"

@st.cache_resource(show_spinner="Loading model …")
def load_model(path=MODEL_PATH):
    if Path(path).exists():
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment = load_model()

with st.sidebar:
    st.header("⚙️ Options")
    conf_cutoff = st.slider("Minimum confidence to accept prediction", 0.50, 1.0, 0.70, 0.01)
    show_wordcloud = st.checkbox("Generate word‑cloud of NEGATIVE reviews", value=True)

col1, col2 = st.columns([2, 1])
with col1:
    text_input = st.text_area("Paste reviews here (one per line):", height=150)
with col2:
    uploaded_file = st.file_uploader("…or upload a CSV/TXT file", type=["csv", "txt"])

reviews = []
if text_input.strip():
    reviews.extend([r.strip() for r in text_input.splitlines() if r.strip()])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_up = pd.read_csv(uploaded_file)
        # Try first column; let user pick otherwise
        col = st.selectbox("Select column containing review text", df_up.columns.tolist())
        reviews.extend(df_up[col].astype(str).tolist())
    else:
        reviews.extend([l.strip() for l in io.StringIO(uploaded_file.getvalue().decode()).read().splitlines() if l.strip()])

if st.button("Analyze reviews") and reviews:
    start = time.time()
    profanity.load_censor_words()
    clean_reviews = [profanity.censor(r) for r in reviews]

    results = sentiment(clean_reviews, truncation=True, batch_size=32)
    runtime = time.time() - start

    df = pd.DataFrame({"review": clean_reviews, "label": [r["label"] for r in results], "score": [r["score"] for r in results]})
    df = df[df.score >= conf_cutoff]

    pos_rate = (df.label == "POSITIVE").mean()
    st.metric("👍 Positive % (filtered)", f"{100*pos_rate:.1f}%")
    st.caption(f"Processed {len(df)} / {len(reviews)} reviews in {runtime:.2f} s (batch_size=32)")

    csv = df.to_csv(index=False).encode()
    st.download_button("Download predictions CSV", csv, "review_predictions.csv", "text/csv")

    st.dataframe(df, use_container_width=True)

    if show_wordcloud and not df[df.label == "NEGATIVE"].empty:
        neg_text = " ".join(df[df.label == "NEGATIVE"].review.tolist())
        wc = WordCloud(width=800, height=400, collocations=False, background_color="white").generate(neg_text)
        st.subheader("Most frequent words in negative reviews")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
