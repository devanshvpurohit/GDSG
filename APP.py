import streamlit as st
import fitz
import docx
import os
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Load LEGAL-BERT model: pile-of-law/legalbert-large-1.7M-2
@st.cache_resource
def load_legalbert_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("pile-of-law/legalbert-large-1.7M-2")
    model = AutoModelForSequenceClassification.from_pretrained("pile-of-law/legalbert-large-1.7M-2")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=5)

legalbert_pipeline = load_legalbert_pipeline()

# Extract text
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

# Analyze with LegalBERT
def analyze_with_legalbert(text):
    # Hugging Face models are limited in input length, we trim to 512 tokens
    return legalbert_pipeline(text[:512])

# Analyze with Gemini Flash
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
st.set_page_config(page_title="📄 AI Content Rights & Licensing Analyzer", layout="wide")
st.title("📄 AI-Powered Content Rights & Licensing Manager")
st.write("Upload a contract file below. We'll extract, analyze, and summarize legal risks and key clauses using LegalBERT and Gemini Flash.")

file = st.file_uploader("📁 Upload Contract (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("📄 Extracting contract text..."):
        contract_text = extract_text(file)

    st.subheader("📃 Extracted Contract Text")
    st.text_area("Text Preview", contract_text, height=300)

    if st.button("🔍 Run AI Analysis"):
        with st.spinner("⚖️ Classifying risks using LegalBERT..."):
            legalbert_results = analyze_with_legalbert(contract_text)

        with st.spinner("🧠 Generating compliance report with Gemini Flash..."):
            gemini_results = analyze_with_gemini(contract_text)

        st.markdown("### ⚠️ Risk Classification (LEGALBERT)")
        st.json(legalbert_results)

        st.markdown("### 📋 Gemini Compliance & Risk Report")
        st.markdown(gemini_results)
