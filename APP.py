import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import google.generativeai as genai
import re
import json
from datetime import datetime

# ---------------------------
# Project: LexiGuardAI
# For: Google Solution Challenge 2025
# Track: Content Rights Management for OTT Platforms (e.g., Aha)
# Target: CXS50 Harvard Offline Students (Avg. GPA 4.5)
# Objective: Automate content licensing, risk assessment, and compliance tracking
# Designed to reflect coding practices of top 7% global developers
# ---------------------------

st.set_page_config(page_title="LexiGuardAI - AI Contract Analyzer", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in environment variables.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Utility: Extract text from uploaded contract
@st.cache_data(show_spinner=False)
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

# Utility: Clean and split into meaningful sentences
def split_sentences(text):
    text = re.sub(r"\n+", " ", text)
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 5]

# Prompt engineering to simulate LegalBERT behavior with Gemini
LEGAL_ANALYSIS_PROMPT = """
You are a legal AI assistant trained to review contracts.
Analyze each sentence for:
- clause category (e.g., Term, Territory, Termination, IP Rights, Indemnity)
- risk level (Low, Medium, High)
- reason for the risk level
Return ONLY valid JSON array: [{"sentence": str, "category": str, "risk": str, "reason": str}, ...]
"""

# Gemini-powered clause analysis
def analyze_sentences_with_gemini(sentences):
    input_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    full_prompt = LEGAL_ANALYSIS_PROMPT + "\n" + input_block
    response = model.generate_content(full_prompt)
    try:
        json_data = json.loads(response.text)
        return json_data
    except Exception:
        return []

# Risk evaluation system for final summary rating
def evaluate_overall_risk(clause_data):
    try:
        risk_score = {"Low": 1, "Medium": 2, "High": 3}
        total = sum(risk_score.get(item.get("risk", "Low"), 1) for item in clause_data)
        avg = total / len(clause_data) if clause_data else 1

        if avg < 1.5:
            return "✅ Low Risk  | ⭐⭐⭐⭐⭐", len(clause_data)
        elif avg < 2.2:
            return "⚠️ Medium Risk  | ⭐⭐⭐✩✩", len(clause_data)
        else:
            return "❌ High Risk  | ⭐✩✩✩✩", len(clause_data)
    except Exception as e:
        return f"Error parsing risk: {e}", 0

# Full summary prompt to Gemini
FULL_DOC_PROMPT = """
You are an expert legal contract assistant.
Provide a professional markdown summary including:
1. Licensing Terms (duration, territory, platforms)
2. Ambiguous Clauses
3. Legal Risks and Violations
4. Actionable Recommendations
5. Executive Summary for Business Teams

--- CONTRACT START ---
"""

# Generate high-level business summary
def analyze_full_contract(text):
    response = model.generate_content(FULL_DOC_PROMPT + text)
    return response.text.strip()

# Optional webhook or Slack/email alert
def send_alert_if_critical(rating):
    if "High Risk" in rating:
        # Placeholder for Slack/Email webhook integration
        timestamp = datetime.now().isoformat()
        print(f"ALERT [{timestamp}]: High Risk Contract Detected")

# Streamlit App Layout
st.title("📄 LexiGuardAI - Rights & Licensing Analyzer")
st.markdown("AI-driven compliance assistant for OTT platforms like Aha.")

file = st.file_uploader("📁 Upload a contract file", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("Extracting contract text..."):
        contract_text = extract_text(file)

    if st.button("🔍 Analyze Contract"):
        with st.spinner("Analyzing clauses using AI..."):
            sentences = split_sentences(contract_text)
            clause_data = analyze_sentences_with_gemini(sentences)
            overall_rating, reviewed_count = evaluate_overall_risk(clause_data)
            send_alert_if_critical(overall_rating)

        with st.spinner("Summarizing the contract..."):
            summary_report = analyze_full_contract(contract_text)

        st.markdown("### 📊 Compliance Summary")
        st.success(f"**{overall_rating}** ({reviewed_count} clauses reviewed)")

        st.markdown("### 📋 Executive Summary")
        st.markdown(summary_report)

        # Future Metrics Dashboard Placeholder
        with st.expander("📈 Show Metrics Dashboard (Coming Soon)"):
            st.info("Clause heatmap, risk trend, and audit trail coming in v2.0")

# Footer
st.markdown("---")
st.caption("LexiGuardAI | Google Solution Challenge 2025")
