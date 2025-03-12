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
import numpy as np

st.set_page_config(page_title="LexiGuardAI v2.0 - AI Contract Analyzer", layout="wide")

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
            return "✅ Low Risk", 5
        elif avg < 2.2:
            return "⚠️ Medium Risk", 3
        else:
            return "❌ High Risk", 1
    except Exception as e:
        return f"Error parsing risk: {e}", 0

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

# Visualization Functions
def generate_risk_heatmap(clause_data):
    if not clause_data:
        return None
    df = pd.DataFrame(clause_data)
    valid_risks = ['High', 'Medium', 'Low']
    df['risk'] = df['risk'].apply(lambda x: x if x in valid_risks else 'Low')
    risk_matrix = pd.crosstab(df['category'], df['risk']).reindex(columns=valid_risks, fill_value=0)
    risk_weights = {'High': 3, 'Medium': 2, 'Low': 1}
    risk_matrix['score'] = risk_matrix['High']*3 + risk_matrix['Medium']*2 + risk_matrix['Low']*1
    risk_matrix = risk_matrix.sort_values('score', ascending=False).drop(columns=['score'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(risk_matrix, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_title("Clause Risk Heatmap")
    ax.set_ylabel("Clause Category")
    ax.set_xlabel("Risk Level")
    plt.tight_layout()
    return fig

def generate_risk_trends(clause_data):
    if not clause_data:
        return None
    df = pd.DataFrame(clause_data)
    valid_risks = ['High', 'Medium', 'Low']
    df['risk'] = df['risk'].apply(lambda x: x if x in valid_risks else 'Low')
    risk_counts = df['risk'].value_counts().reindex(valid_risks, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']
    ax.bar(risk_counts.index, risk_counts.values, color=colors)
    ax.set_title("Risk Distribution Overview")
    ax.set_ylabel("Number of Clauses")
    ax.set_xlabel("Risk Level")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(risk_counts):
        ax.text(i, v + 0.1, str(v), ha='center')
    plt.tight_layout()
    return fig

def categorized_risk_chart(clause_data):
    if not clause_data:
        return None
    df = pd.DataFrame(clause_data)
    valid_risks = ['High', 'Medium', 'Low']
    df['risk'] = df['risk'].apply(lambda x: x if x in valid_risks else 'Low')
    top_categories = df['category'].value_counts().nlargest(8).index.tolist()
    filtered_df = df[df['category'].isin(top_categories)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='category', hue='risk',
                  hue_order=valid_risks,
                  palette={'High': '#d9534f', 'Medium': '#f0ad4e', 'Low': '#5cb85c'},
                  ax=ax)
    ax.set_title("Risk Distribution by Clause Category")
    ax.set_xlabel("Clause Category")
    ax.set_ylabel("Number of Clauses")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title="Risk Level")
    plt.tight_layout()
    return fig

# UI Components
st.title("📄 LexiGuardAI v2.0 - Rights & Licensing Analyzer")
st.markdown("AI-driven compliance assistant for OTT platforms like Aha.")

# Sidebar
st.sidebar.markdown("### 🚀 **v2.0**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.markdown("✅ Contract Analysis\n✅ Risk Assessment\n✅ Clause Heatmap\n✅ Risk Trends\n✅ Audit Trail")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 LexiGuardAI")

# Main App
file = st.file_uploader("📁 Upload a contract file", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("Extracting contract text..."):
        contract_text = extract_text(file)

    if st.button("🔍 Analyze Contract"):
        with st.spinner("Analyzing clauses using AI..."):
            sentences = split_sentences(contract_text)
            clause_data = analyze_sentences_with_gemini(sentences)
            overall_rating, stars = evaluate_overall_risk(clause_data)
            send_alert_if_critical(overall_rating)

        with st.spinner("Generating comprehensive report..."):
            summary_report = analyze_full_contract(contract_text)

        st.markdown("### 📊 Compliance Summary")
        stars_display = "⭐" * stars + "✩" * (5 - stars)
        st.success(f"**{overall_rating} | {stars_display}**")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### 📋 Executive Summary")
            st.markdown(summary_report)

        with col2:
            st.markdown("### 📈 Risk Distribution")
            fig_risk_trend = generate_risk_trends(clause_data)
            if fig_risk_trend:
                st.pyplot(fig_risk_trend)

        st.markdown("### 🔥 Clause Risk Heatmap")
        fig_heatmap = generate_risk_heatmap(clause_data)
        if fig_heatmap:
            st.pyplot(fig_heatmap)

        st.markdown("### 📊 Category Risk Analysis")
        fig_category_risk = categorized_risk_chart(clause_data)
        if fig_category_risk:
            st.pyplot(fig_category_risk)

        with st.expander("📑 Detailed Clause-by-Clause Review", expanded=False):
            if clause_data:
                for idx, item in enumerate(clause_data, 1):
                    st.markdown(f"""
                    **{idx}. {item.get('category', 'Uncategorized')}**  
                    📝 *Clause:* {item['sentence']}  
                    🔐 *Risk Level:* `{item['risk']}`  
                    💡 *Reason:* {item['reason']}
                    """)
                    st.markdown("---")
            else:
                st.warning("No clause-level data available")

st.markdown("---")
st.caption("LexiGuardAI v2.0 | Google Solution Challenge 2025")
