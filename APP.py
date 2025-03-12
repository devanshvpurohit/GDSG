import streamlit as st
import fitz  # PyMuPDF
import docx
import os
import google.generativeai as genai
import re
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="LexiGuardAI - AI Contract Analyzer", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in environment variables.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

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

def split_sentences(text):
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip().split()) > 4]

LEGAL_ANALYSIS_PROMPT = """
You are a legal AI assistant trained to review contracts.
Analyze each sentence for:
- clause category (e.g., Term, Territory, Termination, IP Rights, Indemnity)
- risk level (Low, Medium, High)
- reason for the risk level
Return ONLY valid JSON array: [{"sentence": str, "category": str, "risk": str, "reason": str}, ...]
"""

def analyze_sentences_with_gemini(sentences):
    input_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    full_prompt = LEGAL_ANALYSIS_PROMPT + "\n" + input_block
    response = model.generate_content(full_prompt)
    try:
        json_data = json.loads(response.text)
        return json_data
    except Exception:
        return []

def evaluate_overall_risk(clause_data):
    try:
        risk_score = {"Low": 1, "Medium": 2, "High": 3}
        total = sum(risk_score.get(item.get("risk", "Low"), 1) for item in clause_data)
        clause_count = len(clause_data)
        avg = total / clause_count if clause_count else 1

        if avg < 1.5:
            return "✅ Low Risk", clause_count, 5
        elif avg < 2.2:
            return "⚠️ Medium Risk", clause_count, 3
        else:
            return "❌ High Risk", clause_count, 1
    except Exception as e:
        return f"Error parsing risk: {e}", 0, 0

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

def analyze_full_contract(text):
    response = model.generate_content(FULL_DOC_PROMPT + text)
    return response.text.strip()

def send_alert_if_critical(rating):
    if "High Risk" in rating:
        timestamp = datetime.now().isoformat()
        print(f"ALERT [{timestamp}]: High Risk Contract Detected")

def generate_heatmap_and_trends(clause_data):
    df = pd.DataFrame(clause_data)
    if df.empty:
        return None, None

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='category', hue='risk', palette='coolwarm', ax=ax1)
    ax1.set_title("Clause Risk Distribution by Category")
    ax1.set_ylabel("Clause Count")
    plt.xticks(rotation=45)

    risk_timeline = pd.DataFrame({"timestamp": [datetime.now()] * len(df), "risk": df["risk"]})
    risk_timeline_grouped = risk_timeline.groupby("risk").count()

    fig2, ax2 = plt.subplots()
    risk_timeline_grouped.plot(kind='bar', legend=False, ax=ax2, color='orange')
    ax2.set_title("Risk Frequency Overview")
    ax2.set_ylabel("Occurrences")

    return fig1, fig2

def display_audit_trail(clause_data):
    st.markdown("### 🧾 Audit Trail")
    audit_df = pd.DataFrame(clause_data)
    audit_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(audit_df[['category', 'risk', 'reason', 'timestamp']])

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
            overall_rating, reviewed_count, stars = evaluate_overall_risk(clause_data)
            send_alert_if_critical(overall_rating)

        with st.spinner("Summarizing the contract..."):
            summary_report = analyze_full_contract(contract_text)

        st.markdown("### 📊 Compliance Summary")
        stars_display = "⭐" * stars + "✩" * (5 - stars)
        st.success(f"**{overall_rating} | {stars_display}**  ({reviewed_count} clauses reviewed)")

        st.markdown("### 📋 Executive Summary")
        st.markdown(summary_report)

        st.markdown("### 📑 Clause-by-Clause Review")
        if clause_data:
            for idx, item in enumerate(clause_data, 1):
                st.markdown(f"**{idx}. {item.get('category', 'Uncategorized')}**")
                st.write(f"📝 *Clause:* {item['sentence']}")
                st.write(f"🔐 *Risk Level:* `{item['risk']}`")
                st.write(f"💡 *Reason:* {item['reason']}")
                st.markdown("---")
        else:
            st.warning("No clause-level data returned. Check contract formatting or try again.")

        st.markdown("### 🔥 Clause Heatmap and Risk Trend")
        heatmap_fig, trend_fig = generate_heatmap_and_trends(clause_data)
        if heatmap_fig and trend_fig:
            st.pyplot(heatmap_fig)
            st.pyplot(trend_fig)

        display_audit_trail(clause_data)

st.markdown("---")
st.caption("LexiGuardAI | Google Solution Challenge 2025")
