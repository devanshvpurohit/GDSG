import streamlit as st
import requests
import google.generativeai as genai

# --- API Keys (replace with actual keys directly here) ---
HF_API_KEY = "YOUR_HUGGINGFACE_API_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# --- Hugging Face LEGAL-BERT API Setup ---
LEGAL_BERT_API_URL = "https://api-inference.huggingface.co/models/nlpaueb/legal-bert-base-uncased"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- Functions to Process Contracts ---

# Function to analyze contract using LEGAL-BERT
def analyze_contract(text):
    response = requests.post(LEGAL_BERT_API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to analyze contract. Error code: {response.status_code}"}

# Function to get compliance summary using Gemini API
def get_compliance_summary(text):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        f"Analyze this licensing contract for compliance issues, summarize key licensing terms (like duration, regions, platforms), and highlight any ambiguous clauses:\n\n{text}"
    )
    return response.text

# --- Streamlit App UI ---

# Title
st.set_page_config(page_title="Content Rights & Licensing Manager", page_icon="📜")
st.title("📜 AI-Powered Content Rights & Licensing Manager")

# File Upload
uploaded_file = st.file_uploader("Upload a contract file (TXT format only)", type=["txt"])

# Analyze Button and Result Display
if uploaded_file:
    contract_text = uploaded_file.read().decode("utf-8")
    st.subheader("📄 Contract Preview")
    st.text_area("Contract Content", contract_text, height=300)

    if st.button("🔍 Analyze Contract"):
        with st.spinner("Analyzing the contract using AI models..."):
            # AI-based Analysis
            bert_analysis = analyze_contract(contract_text)
            gemini_summary = get_compliance_summary(contract_text)

        # Display Results
        st.subheader("📑 LEGAL-BERT Contract Analysis")
        if isinstance(bert_analysis, dict) and "error" in bert_analysis:
            st.error(bert_analysis["error"])
        else:
            st.json(bert_analysis)

        st.subheader("✅ Gemini AI Compliance Summary")
        st.write(gemini_summary)

        # Simple Risk Warning based on LEGAL-BERT (Optional)
        if isinstance(bert_analysis, list):
            risk_labels = [item['label'] for item in bert_analysis]
            risk_scores = [item['score'] for item in bert_analysis if item['label'] == "Non-Compliant"]
            if risk_scores and risk_scores[0] > 0.5:
                st.error("🚨 Warning: This contract might have compliance issues. Please review carefully!")
            else:
                st.success("✅ The contract appears compliant based on AI analysis.")
