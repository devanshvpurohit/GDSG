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
import io


import streamlit.components.v1 as components

# Set Streamlit Page Config
st.set_page_config(page_title="LexiGuardAI v2.0 - AI Contract Analyzer", layout="wide")

# HTML + JavaScript for the Interactive Constellation Background
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Interactive Constellation</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body, html { width: 100%; height: 100%; overflow: hidden; background-color: black; }
    canvas { display: block; position: fixed; top: 0; left: 0; z-index: -1; }
  </style>
</head>
<body>
<canvas id="constellation"></canvas>
<script>
  const canvas = document.getElementById('constellation');
  const ctx = canvas.getContext('2d');

  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;

  const dots = [];
  const dotCount = 100;
  const maxDist = 150;
  const mouseRadius = 200;
  let mouse = { x: null, y: null };

  // Create dots
  for (let i = 0; i < dotCount; i++) {
    dots.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5
    });
  }

  // Mouse move event
  canvas.addEventListener('mousemove', function (e) {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  // Resize event
  window.addEventListener('resize', function () {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  // Draw function
  function draw() {
    ctx.clearRect(0, 0, width, height);

    // Draw dots
    for (let dot of dots) {
      ctx.beginPath();
      ctx.arc(dot.x, dot.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'red';
      ctx.fill();
    }

    // Connect dots near the mouse
    for (let i = 0; i < dots.length; i++) {
      for (let j = i + 1; j < dots.length; j++) {
        let dx = dots[i].x - dots[j].x;
        let dy = dots[i].y - dots[j].y;
        let distance = Math.sqrt(dx * dx + dy * dy);

        let mouseDx = dots[i].x - mouse.x;
        let mouseDy = dots[i].y - mouse.y;
        let mouseDistance = Math.sqrt(mouseDx * mouseDx + mouseDy * mouseDy);

        if (distance < maxDist && mouseDistance < mouseRadius) {
          ctx.beginPath();
          ctx.moveTo(dots[i].x, dots[i].y);
          ctx.lineTo(dots[j].x, dots[j].y);
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
          ctx.stroke();
        }
      }
    }

    // Move dots
    for (let dot of dots) {
      dot.x += dot.vx;
      dot.y += dot.vy;

      // Bounce off walls
      if (dot.x <= 0 || dot.x >= width) dot.vx *= -1;
      if (dot.y <= 0 || dot.y >= height) dot.vy *= -1;
    }

    requestAnimationFrame(draw);
  }

  draw();
</script>
</body>
</html>
"""

# Inject HTML into Streamlit (as a fixed background)
components.html(html_code, height=0, width=0)


# API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in environment variables.")
    st.stop()

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
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip().split()) > 4]

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
    response = model.generate_content(LEGAL_ANALYSIS_PROMPT + "\n" + input_block)
    try:
        return json.loads(response.text)
    except:
        return []

def evaluate_overall_risk(clause_data):
    risk_score = {"Low": 1, "Medium": 2, "High": 3}
    total = sum(risk_score.get(item.get("risk", "Low"), 1) for item in clause_data)
    avg = total / len(clause_data) if clause_data else 1
    if avg < 1.5: return "✅ Low Risk", 5
    elif avg < 2.2: return "⚠️ Medium Risk", 3
    else: return "❌ High Risk", 1

def generate_risk_heatmap(clause_data):
    df = pd.DataFrame(clause_data)
    if df.empty: return None
    df['risk'] = df['risk'].apply(lambda x: x if x in ['High', 'Medium', 'Low'] else 'Low')
    matrix = pd.crosstab(df['category'], df['risk']).reindex(columns=['High', 'Medium', 'Low'], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.tight_layout(); plt.close(fig)
    return fig

def generate_risk_trends(clause_data):
    df = pd.DataFrame(clause_data)
    if df.empty: return None
    counts = df['risk'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=['#d9534f', '#f0ad4e', '#5cb85c'])
    plt.tight_layout(); plt.close(fig)
    return fig

def categorized_risk_chart(clause_data):
    df = pd.DataFrame(clause_data)
    if df.empty: return None
    df['risk'] = df['risk'].apply(lambda x: x if x in ['High', 'Medium', 'Low'] else 'Low')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='category', hue='risk', order=df['category'].value_counts().index[:8], ax=ax)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.close(fig)
    return fig

# UI
st.title("📄 LexiGuardAI v2.0 - Rights & Licensing Analyzer")
st.markdown("AI-driven compliance assistant for OTT platforms like Aha.")

file = st.file_uploader("📁 Upload a contract file", type=["pdf", "docx", "txt"])

if file:
    contract_text = extract_text(file)
    if st.button("🔍 Analyze Contract"):
        sentences = split_sentences(contract_text)
        clause_data = analyze_sentences_with_gemini(sentences)
        overall_rating, stars = evaluate_overall_risk(clause_data)

        st.markdown(f"### 📊 Compliance Summary\n**{overall_rating} {'⭐'*stars + '✩'*(5-stars)}**")

        # Visualizations
        for title, func in zip(["🔥 Clause Risk Heatmap", "📈 Risk Trends Overview", "📊 Category-wise Clause Risk"],
                               [generate_risk_heatmap, generate_risk_trends, categorized_risk_chart]):
            st.markdown(f"### {title}")
            fig = func(clause_data)
            if fig: st.pyplot(fig)
            else: st.info("No sufficient data for visualization.")

        # Audit Trail
        st.markdown("### 🗂️ Clause Audit Trail")
        audit_df = pd.DataFrame(clause_data)
        csv_buffer, json_buffer = io.StringIO(), io.StringIO()
        audit_df.to_csv(csv_buffer, index=False)
        json.dump(clause_data, json_buffer, indent=4)
        st.download_button("⬇️ Download Clause Audit (CSV)", csv_buffer.getvalue(), "contract_clause_audit.csv", "text/csv")
        st.download_button("⬇️ Download Clause Audit (JSON)", json_buffer.getvalue(), "contract_clause_audit.json", "application/json")

        # Clause Review
        with st.expander("📑 Detailed Clause-by-Clause Review", expanded=False):
            for idx, item in enumerate(clause_data, 1):
                st.markdown(f"""
                **{idx}. {item.get('category', 'Uncategorized')}**  
                📝 *Clause:* {item['sentence']}  
                🔐 *Risk Level:* `{item['risk']}`  
                💡 *Reason:* {item['reason']}
                """)
                st.markdown("---")

st.caption("LexiGuardAI v2.0 | Google Solution Challenge 2025")
