import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import requests
import google.generativeai as genai

# ✅ MUST BE FIRST
st.set_page_config(page_title="📄 AI Content Rights & Licensing Analyzer", layout="wide")

# 🔐 Load API keys from environment (recommended for Streamlit Cloud)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", st.secrets.get("HF_API_TOKEN", ""))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ✅ Check keys
if not HF_API_TOKEN:
    st.error("🚨 Hugging Face API key not found. Set HF_API_TOKEN in environment.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("🚨 Gemini API key not found. Set GEMINI_API_KEY in environment.")
    st.stop()

# ✅ Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# ✅ Extract contract text
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

# ✅ Analyze with Hugging Face LegalBERT (via API)
def analyze_with_legalbert(text):
    url = "https://api-inference.huggingface.co/models/pile-of-law/legalbert-large-1.7M-2"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {
        "inputs": text[:512]  # Truncate to avoid model input length errors
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return {"error": f"Hugging Face API error: {response.status_code} {response.text}"}
    return response.json()

# ✅ Analyze with Gemini
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

# ✅ Streamlit App UI
st.title("📄 AI-Powered Content Rights & Licensing Manager")
st.markdown("Upload a contract file to extract legal terms, assess risk, and get revision suggestions.")

file = st.file_uploader("📁 Upload Contract File (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("📄 Extracting text..."):
        contract_text = extract_text(file)

    st.subheader("📃 Contract Preview")
    st.text_area("Extracted Text", contract_text, height=300)

    if st.button("🔍 Run AI Analysis"):
        with st.spinner("⚖️ Analyzing with LegalBERT (Hugging Face)..."):
            legalbert_results = analyze_with_legalbert(contract_text)

        with st.spinner("🧠 Generating report with Gemini Flash..."):
            gemini_results = analyze_with_gemini(contract_text)

        st.markdown("### ⚠️ Legal Risk Classification (LegalBERT - Hugging Face)")
        st.json(legalbert_results)

        st.markdown("### 📋 Gemini Risk & Licensing Summary")
        st.markdown(gemini_results)
