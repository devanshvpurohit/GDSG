import streamlit as st
import fitz  # PyMuPDF
import docx
import requests
import os

# Load API keys from environment variables or Streamlit Cloud secrets
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", st.secrets.get("HUGGINGFACE_API_KEY", ""))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# File parsing
def extract_text(file):
    ext = file.name.lower()
    try:
        if ext.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif ext.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext.endswith(".txt"):
            return file.read().decode("utf-8")
        else:
            return "Unsupported file type."
    except Exception as e:
        return f"Error during text extraction: {str(e)}"

# Hugging Face LEGAL-BERT API
def classify_risks_hf(text):
    url = "https://api-inference.huggingface.co/models/nlpaueb/legal-bert-base-uncased"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text[:512]}  # Limit input length
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e), "details": response.text}

# Gemini Pro API
def query_gemini(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    prompt = f"""
Analyze the following licensing contract for compliance and risk issues:

{text}

Provide a structured markdown report with:
1. Licensing Terms (duration, territory, platforms)
2. Ambiguous Clauses
3. Legal Risks or Violations
4. Actionable Recommendations
5. Summary for Business Teams
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }

    response = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload)
    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error from Gemini: {e}\n\nRaw response:\n{response.text}"

# Streamlit UI
st.set_page_config(page_title="AI Licensing Manager", layout="wide")
st.title("📄 AI-Powered Content Rights & Licensing Manager")
st.write("Upload a licensing contract to extract key terms, detect risks, and get recommendations.")

uploaded_file = st.file_uploader("Upload Contract (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        contract_text = extract_text(uploaded_file)

    st.subheader("📃 Contract Preview")
    st.text_area("Contract Text", contract_text, height=300)

    if st.button("🔍 Run AI Analysis"):
        with st.spinner("Analyzing contract..."):
            risk_result = classify_risks_hf(contract_text)
            gemini_report = query_gemini(contract_text)

        st.markdown("### ⚠️ Risk Classification (LEGAL-BERT via Hugging Face)")
        st.json(risk_result)

        st.markdown("### 📋 Compliance Report (Gemini Pro)")
        st.markdown(gemini_report)
