import streamlit as st
import pickle
import re

# ✅ FIRST Streamlit command — must come before anything else
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Inject the local CSS file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# Load model and vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)
st.markdown("""
<div style='
    background-color: #333;
    padding: 15px;
    border-radius: 10px;
    color: #f0f0f0;
    font-size: 16px;
    margin-bottom: 20px;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
'>
    <strong>📝 Note:</strong> To help the analyzer work properly, please enter <u>at least 1–2 paragraphs</u> of a news article. The more content you provide, the better the model can assess it.
</div>
""", unsafe_allow_html=True)



# Preprocess input
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

# Title with custom styles
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
            st.markdown(f'<div class="result negative">❌ This is likely **FAKE NEWS** ({prob[0]*100:.2f}% confidence)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result positive">✅ This is likely **REAL NEWS** ({prob[1]*100:.2f}% confidence)</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Developed By Moeez Nabi Wani – Check the credibility of news articles in seconds!</div>", unsafe_allow_html=True)
