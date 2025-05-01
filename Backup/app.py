import streamlit as st
import pickle
import re

# Load model and vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

    

# Preprocess input
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

# Set up Streamlit page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Title with custom styles
st.markdown("""
    <style>
        .title {
            font-family: 'Arial', sans-serif;
            text-align: center;
            color: #2c3e50;
            font-size: 36px;
            margin-top: 20px;
            animation: fadeIn 2s ease-in-out;
        }
        
        .stTextArea>div>div>textarea {
            border: 2px solid #3498db;
            border-radius: 8px;
            font-size: 16px;
            padding: 15px;
        }

        .stButton>button {
            background-color: #3498db;
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 12px 28px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
        }

        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        .content {
            font-family: 'Arial', sans-serif;
            color: #34495e;
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
        }

        .result {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
        }

        .positive {
            color: green;
        }

        .negative {
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

# Display Title
st.markdown('<div class="title">Fake News Detection</div>', unsafe_allow_html=True)

# News Input Section
news_input = st.text_area("Enter News Text Below:")

# Button to analyze
if st.button("Analyze"):
    if news_input.strip() == "":
        st.warning("Please enter some news content to analyze.")
    else:
        cleaned = clean(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        # Displaying results with animation and color
        if prediction == 0:
            st.markdown(f'<p class="result negative">❌ This is likely **FAKE NEWS** ({prob[0]*100:.2f}% confidence)</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="result positive">✅ This is likely **REAL NEWS** ({prob[1]*100:.2f}% confidence)</p>', unsafe_allow_html=True)

# Additional footer or content (optional)
st.markdown("<div class='content'>Developed By Moeez Nabi Wani – Check the credibility of news articles in seconds!</div>", unsafe_allow_html=True)
