import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Function to load the pre-trained model
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)
    return sentiment_pipeline

# Streamlit app
st.title("Basic Sentiment Analysis App -- loaded from hugging-face spaces ")
st.write("Enter a text and select a pre-trained model to get the sentiment analysis.")

# Input text
default_text = "I love my life."
text = st.text_input("Enter your text:", value=default_text)

# Model selection
model_options = {
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "labels": ["NEGATIVE", "POSITIVE"],
        "description": "This model classifies text into positive or negative sentiment. It is based on DistilBERT and fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset.",
    },
    "textattack/bert-base-uncased-SST-2": {
        "labels": ["LABEL_0", "LABEL_1"],
        "description": "This model classifies text into positive(LABEL_1) or negative(LABEL_0) sentiment. It is based on BERT and fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset.",
    },
    "cardiffnlp/twitter-roberta-base-sentiment": {
        "labels": ["LABEL_0", "LABEL_1", "LABEL_2"],
        "description": "This model classifies tweets into negative (LABEL_0), neutral(LABEL_1), or positive(LABEL_2) sentiment. It is based on RoBERTa and fine-tuned on a large dataset of tweets.",
    },
}
selected_model = st.selectbox("Choose a pre-trained model:", model_options)

st.write("### Model Information")
st.write(f"**Labels:** {', '.join(model_options[selected_model]['labels'])}")
st.write(f"**Description:** {model_options[selected_model]['description']}")

# Load the model and perform sentiment analysis
if st.button("Analyze"):
    if not text:
        st.write("Please enter a text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            sentiment_pipeline = load_model(selected_model)
            result = sentiment_pipeline(text)
            st.write(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.2f})")
            if result[0]['label'] in ['POSITIVE', 'LABEL_1'] and result[0]['score']> 0.9:
                st.balloons()
            elif result[0]['label'] in ['NEGATIVE', 'LABEL_0'] and result[0]['score']> 0.9:
                st.error("Hater detected.")
else:
    st.write("Enter a text and click 'Analyze' to perform sentiment analysis.")