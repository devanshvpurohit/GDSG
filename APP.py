import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import requests
import google.generativeai as genai
import re

# Set Streamlit config
st.set_page_config(page_title="📄 AI Licensing Manager (Sentence-Level)", layout="wide")

# API keys
HF_API_TOKEN = os.getenv("HF_API_TOKEN", st.secrets.get("HF_API_TOKEN", ""))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

if not HF_API_TOKEN or not GEMINI_API_KEY:
    st.error("Missing API keys. Set HF_API_TOKEN and GEMINI_API_KEY.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Extract text from supported formats
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")
        else:
            return "Unsupported file format."
    except Exception as e:
        return f"Error extracting text: {e}"

# Break contract into clean sentences
def split_sentences(text):
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

# Analyze each sentence via Hugging Face LegalBERT
def analyze_sentences(sentences):
    url = "https://api-inference.huggingface.co/models/pile-of-law/legalbert-large-1.7M-2"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    results = []

    for sentence in sentences:
        payload = {"inputs": sentence}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            results.append({
                "sentence": sentence,
                "labels": result[0] if isinstance(result, list) else []
            })
        else:
            results.append({
                "sentence": sentence,
                "labels": [{"label": "API_ERROR", "score": 0.0}]
            })
    return results

# Use Gemini Flash for full document analysis
def analyze_with_gemini(text):
    prompt = f"""
You're a legal AI assistant. Analyze the following licensing contract and return a markdown report with:

1. Licensing Terms (duration, territory, platforms)
2. Ambiguous Clauses
3. Legal Risks or Violations
4. Actionable Recommendations
5. Summary for Business Teams

Contract:
{text}
"""
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📄 AI-Powered Licensing & Contract Risk Analyzer")
st.markdown("Upload a contract. We’ll give you sentence-by-sentence risk review + an AI compliance summary.")

file = st.file_uploader("📁 Upload Contract (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("📄 Extracting text..."):
        contract_text = extract_text(file)

    st.subheader("📃 Contract Preview")
    st.text_area("Extracted Contract", contract_text, height=300)

    if st.button("🔍 Run Sentence-Level AI Analysis"):
        with st.spinner("✂️ Splitting into sentences..."):
            sentences = split_sentences(contract_text)

        with st.spinner("⚖️ Analyzing each sentence with LegalBERT..."):
            sentence_results = analyze_sentences(sentences)

        st.markdown("### 🧩 Per-Sentence Legal Risk Analysis (Hugging Face LegalBERT)")

        for item in sentence_results:
            st.markdown(f"**📝 Sentence:** {item['sentence']}")
            st.json(item["labels"])

        with st.spinner("📊 Generating full document analysis with Gemini Flash..."):
            gemini_summary = analyze_with_gemini(contract_text)

        st.markdown("### 📋 Gemini Compliance Report")
        st.markdown(gemini_summary)
