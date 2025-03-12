import streamlit as st
import requests
import google.generativeai as genai
import fitz  # PyMuPDF for PDF
import docx  # python-docx for Word files

# --- API Keys (replace directly here for testing) ---
HF_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyAW_b4mee9l8eP931cqd9xqErHV34f7OEw"

# --- Hugging Face LEGAL-BERT API Setup ---
LEGAL_BERT_API_URL = "https://api-inference.huggingface.co/models/nlpaueb/legal-bert-base-uncased"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- Functions to Process Files and Contracts ---

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Analyze contract using LEGAL-BERT
def analyze_contract(text):
    response = requests.post(LEGAL_BERT_API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to analyze contract. Error code: {response.status_code}"}

# Get compliance summary using Gemini API
def get_compliance_summary(text):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        f"Analyze this licensing contract for compliance issues. Summarize key licensing terms (like duration, regions, platforms), highlight any ambiguous clauses, and give recommendations for legal compliance:\n\n{text}"
    )
    return response.text

# --- Streamlit App UI ---

# Title
st.set_page_config(page_title="AI Licensing & Compliance Manager", page_icon="📜")
st.title("📜 AI-Powered Licensing & Compliance Manager")

# File Upload (Multiple formats)
uploaded_file = st.file_uploader("Upload a contract file (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

# Analyze Button and Result Display
if uploaded_file:
    file_type = uploaded_file.type
    contract_text = ""

    # Extract text based on file type
    if file_type == "text/plain":
        contract_text = uploaded_file.read().decode("utf-8")
    elif file_type == "application/pdf":
        contract_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        contract_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format!")

    if contract_text:
        st.subheader("📄 Contract Preview")
        st.text_area("Contract Content", contract_text, height=300)

        if st.button("🔍 Analyze Contract"):
            with st.spinner("Analyzing the contract with AI models..."):
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
                    st.error("🚨 Warning: High risk of compliance issues detected. Please review carefully!")
                else:
                    st.success("✅ No major compliance issues detected based on AI analysis.")
