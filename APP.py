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

# V2.0 FEATURES
def generate_risk_heatmap(clause_data):
    """Generate a heatmap showing risk distribution by clause category"""
    if not clause_data:
        return None
    
    df = pd.DataFrame(clause_data)
    
    # Cross tabulation of category vs risk
    risk_matrix = pd.crosstab(df['category'], df['risk'])
    
    # Sort categories by total risk score
    risk_weights = {'Low': 1, 'Medium': 2, 'High': 3}
    risk_matrix['score'] = risk_matrix.get('High', 0) * 3 + risk_matrix.get('Medium', 0) * 2 + risk_matrix.get('Low', 0)
    risk_matrix = risk_matrix.sort_values('score', ascending=False)
    risk_matrix = risk_matrix.drop(columns=['score'])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    if not risk_matrix.empty:
        ax = sns.heatmap(risk_matrix, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5)
        plt.title("Clause Risk Heatmap")
        plt.ylabel("Clause Category")
        plt.xlabel("Risk Level")
        plt.tight_layout()
        return plt.gcf()
    return None

def generate_risk_trends(clause_data):
    """Generate a visualization of risk distribution"""
    if not clause_data:
        return None
    
    df = pd.DataFrame(clause_data)
    
    # Count risks by category
    risk_counts = df['risk'].value_counts().reindex(['High', 'Medium', 'Low'])
    
    plt.figure(figsize=(8, 5))
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']  # Red, Orange, Green
    ax = risk_counts.plot(kind='bar', color=colors)
    
    plt.title("Risk Distribution Overview")
    plt.ylabel("Number of Clauses")
    plt.xlabel("Risk Level")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels on top of bars
    for i, v in enumerate(risk_counts):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def display_audit_trail(clause_data):
    """Generate audit trail data"""
    if not clause_data:
        return None
    
    audit_df = pd.DataFrame(clause_data)
    # Add timestamp and user info
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    audit_df['analyzed_at'] = timestamp
    audit_df['analyzed_by'] = "LexiGuardAI v2.0"
    
    return audit_df[['category', 'risk', 'reason', 'analyzed_at', 'analyzed_by']]

def categorized_risk_chart(clause_data):
    """Create a bar chart showing risk levels by category"""
    if not clause_data:
        return None
        
    df = pd.DataFrame(clause_data)
    # Get top 8 categories by frequency
    top_categories = df['category'].value_counts().nlargest(8).index.tolist()
    filtered_df = df[df['category'].isin(top_categories)]
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=filtered_df, x='category', hue='risk', 
                       hue_order=['High', 'Medium', 'Low'],
                       palette={'High': '#d9534f', 'Medium': '#f0ad4e', 'Low': '#5cb85c'})
    
    plt.title("Risk Distribution by Clause Category")
    plt.xlabel("Clause Category")
    plt.ylabel("Number of Clauses")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Risk Level")
    plt.tight_layout()
    
    return plt.gcf()

# UI COMPONENTS
st.title("📄 LexiGuardAI v2.0 - Rights & Licensing Analyzer")
st.markdown("AI-driven compliance assistant for OTT platforms like Aha.")

# Add a version badge
st.sidebar.markdown("### 🚀 **v2.0**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.markdown("✅ Contract Analysis")
st.sidebar.markdown("✅ Risk Assessment")
st.sidebar.markdown("✅ Clause Heatmap")
st.sidebar.markdown("✅ Risk Trends")
st.sidebar.markdown("✅ Audit Trail")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 LexiGuardAI")

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

        with st.spinner("Summarizing the contract..."):
            summary_report = analyze_full_contract(contract_text)

        st.markdown("### 📊 Compliance Summary")
        stars_display = "⭐" * stars + "✩" * (5 - stars)
        st.success(f"**{overall_rating} | {stars_display}**")

        # Two-column layout for summary and risk visualization
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📋 Executive Summary")
            st.markdown(summary_report)

        with col2:
            st.markdown("### 📈 Risk Distribution")
            risk_trend_fig = generate_risk_trends(clause_data)
            if risk_trend_fig:
                st.pyplot(risk_trend_fig)
            else:
                st.info("No risk data available to visualize.")

        # Clause heatmap
        st.markdown("### 🔥 Clause Risk Heatmap")
        heatmap_fig = generate_risk_heatmap(clause_data)
        if heatmap_fig:
            st.pyplot(heatmap_fig)
        else:
            st.info("Insufficient data to generate heatmap.")
            
        # Category risk distribution
        st.markdown("### 📊 Category Risk Analysis")
        category_fig = categorized_risk_chart(clause_data)
        if category_fig:
            st.pyplot(category_fig)
        else:
            st.info("Insufficient data for category analysis.")

        # Clause-by-clause review in an expandable section
        with st.expander("📑 Detailed Clause-by-Clause Review", expanded=False):
            if clause_data:
                for idx, item in enumerate(clause_data, 1):
                    st.markdown(f"**{idx}. {item.get('category', 'Uncategorized')}**")
                    st.write(f"📝 *Clause:* {item['sentence']}")
                    st.write(f"🔐 *Risk Level:* `{item['risk']}`")
                    st.write(f"💡 *Reason:* {item['reason']}")
                    st.markdown("---")
            else:
                st.warning("No clause-level data returned. Check contract formatting or try again.")

        # Audit trail
        with st.expander("🧾 Audit Trail", expanded=False):
            audit_df = display_audit_trail(clause_data)
            if audit_df is not None:
                st.dataframe(audit_df)
            else:
                st.info("No audit data available.")

st.markdown("---")
st.caption("LexiGuardAI v2.0 | Google Solution Challenge 2025")
