import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import google.generativeai as genai
import re
import json

# ---------------------------
# Project: LexiGuardAI
# For: Google Solution Challenge 2025
# Track: Content Rights Management for OTT Platforms (e.g., Aha)
# Target: CXS50 Harvard Offline Students (Avg. GPA 4.5)
# Objective: Automate content licensing, risk assessment, and compliance tracking
# ---------------------------

# Set Streamlit page config
st.set_page_config(page_title="LexiGuardAI - AI Contract Analyzer", layout="wide")

# Set API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in environment variables.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Extract text from uploaded contract
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

# Clean and split text into sentences
def split_sentences(text):
    text = re.sub(r"\n+", " ", text)
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 5]

# Simulated LegalBERT-like prompt analysis via Gemini
LEGAL_ANALYSIS_PROMPT = """
You are a legal AI assistant trained to analyze contract clauses like LegalBERT.
For each sentence below, return:
- Sentence
- Clause Category (e.g. Term, Territory, Termination, IP Rights, Indemnity)
- Risk Level (Low, Medium, High)
- Reason for Risk Rating
Return output in JSON format as a list of dictionaries with keys: sentence, category, risk, reason.
"""

def analyze_sentences_with_gemini(sentences):
    input_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    full_prompt = LEGAL_ANALYSIS_PROMPT + "\n" + input_block
    response = model.generate_content(full_prompt)
    return response.text

# Parse risk levels and calculate summary score
def evaluate_overall_risk(json_text):
    try:
        data = json.loads(json_text)
        risk_score = {"Low": 1, "Medium": 2, "High": 3}
        total = sum(risk_score.get(item.get("risk", "Low"), 1) for item in data)
        avg = total / len(data) if data else 1
        if avg < 1.5:
            rating = "✅ Low Risk"
            stars = "⭐⭐⭐⭐⭐"
        elif avg < 2.2:
            rating = "⚠️ Medium Risk"
            stars = "⭐⭐⭐✩✩"
        else:
            rating = "❌ High Risk"
            stars = "⭐✩✩✩✩"
        return f"{rating}  | Compliance Score: {stars}", len(data)
    except Exception as e:
        return f"Error parsing JSON: {e}", 0

# Full document summary using Gemini
FULL_DOC_PROMPT = """
You are a legal contract analyst.
Given the contract below, generate a structured markdown report:

1. Licensing Terms (duration, territory, platforms)
2. Ambiguous Clauses
3. Legal Risks and Violations
4. Actionable Recommendations
5. Summary for Business Teams

Contract:
"""

def analyze_full_contract(text):
    prompt = FULL_DOC_PROMPT + text
    response = model.generate_content(prompt)
    return response.text

# Placeholder for future real-time alert webhook (e.g., Slack, Email)
def send_alert_if_critical(rating):
    if "High Risk" in rating:
        # Future: send email/slack alert to legal team
        pass

# Streamlit UI
st.title("📄 LexiGuardAI - AI-Powered Rights & Licensing Analyzer")
st.markdown("Empowering OTT platforms like Aha with AI-driven content contract analysis and compliance.")

file = st.file_uploader("📁 Upload contract file", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("Extracting text from document..."):
        contract_text = extract_text(file)

    if st.button("🔍 Run Legal AI Analysis"):
        with st.spinner("Splitting and analyzing clauses..."):
            sentences = split_sentences(contract_text)
            clause_json_text = analyze_sentences_with_gemini(sentences)
            overall_rating, count = evaluate_overall_risk(clause_json_text)
            send_alert_if_critical(overall_rating)

        with st.spinner("Generating summary report..."):
            full_report = analyze_full_contract(contract_text)

        st.markdown("### 📊 Overall Contract Risk & Compliance Rating")
        st.success(f"**{overall_rating}** ({count} clauses reviewed)")

        st.markdown("### 📋 AI-Generated Summary Report")
        st.markdown(full_report)

# Footer
st.markdown("---")
st.caption("LexiGuardAI • CXS50 Harvard • Google Solution Challenge 2025")
