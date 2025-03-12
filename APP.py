import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import google.generativeai as genai
import re

# Set Streamlit page config
st.set_page_config(page_title="📄 AI Licensing Risk Analyzer", layout="wide")

# Set API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key missing. Set GEMINI_API_KEY as env variable or secret.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Text extraction function
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")
        else:
            return "Unsupported file format."
    except Exception as e:
        return f"Error reading file: {e}"

# Split into clean sentences
def split_sentences(text):
    text = re.sub(r"\n+", " ", text)
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 5]

# Pseudo-fine-tune Gemini via prompt
def analyze_sentences(sentences):
    results = []
    base_prompt = """
You are a legal contract analysis AI assistant trained like LegalBERT.

For each sentence provided, return:
- Sentence
- Legal Category (e.g. Term, Territory, Rights, Indemnity, Exclusivity, Termination, Confidentiality)
- Risk Level (Low, Medium, High)
- Justification (1-2 lines)

Return result in structured JSON format.
"""

    full_prompt = base_prompt + "\nSentences:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    response = model.generate_content(full_prompt)
    return response.text

# Gemini full document summarization
def analyze_contract_full(text):
    prompt = f"""
You are a legal AI assistant.

Analyze this licensing contract and return a markdown report with:

1. Licensing Terms (duration, territory, platforms)
2. Ambiguous Clauses
3. Legal Risks or Violations
4. Actionable Recommendations
5. Business Summary

Contract:
{text}
"""
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📄 AI-Powered Contract Licensing Analyzer (Gemini Only)")
st.markdown("Upload a contract. AI will simulate LegalBERT-like analysis via Gemini and provide detailed sentence-level and full-document review.")

file = st.file_uploader("📁 Upload contract file", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("📄 Extracting contract text..."):
        contract_text = extract_text(file)

    st.subheader("📃 Contract Text Preview")
    st.text_area("Extracted Contract", contract_text, height=300)

    if st.button("🔎 Run AI Analysis"):
        with st.spinner("✂️ Splitting into sentences..."):
            sentences = split_sentences(contract_text)

        with st.spinner("🤖 Analyzing each sentence with Gemini (simulated LegalBERT)..."):
            sentence_analysis = analyze_sentences(sentences)
            st.markdown("### 🧩 Per-Sentence Legal Risk Analysis")
            st.markdown(sentence_analysis)

        with st.spinner("📊 Generating full document compliance report..."):
            full_report = analyze_contract_full(contract_text)
            st.markdown("### 📋 Full Contract Analysis (Gemini Flash)")
            st.markdown(full_report)
