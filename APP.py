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
# Streamlit Config
st.set_page_config(page_title="LexiGuardAI v2.0 - AI Contract Analyzer", layout="wide")

# HTML Animation
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LexiGuardAI Animation</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body, html { width: 100%; height: 100%; overflow: hidden; background: black; }
  canvas { display: block; position: absolute; top: 0; left: 0; }
</style>
</head>
<body>
<canvas id="animationCanvas"></canvas>
<script>
const canvas = document.getElementById('animationCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const particles = [];
const numParticles = 800;

class Particle {
  constructor(x, y, color) {
    this.x = x;
    this.y = y;
    this.color = color;
    this.size = 2.5;
    this.vx = (Math.random() - 0.5) * 2;
    this.vy = (Math.random() - 0.5) * 2;
    this.targetX = null;
    this.targetY = null;
  }
  update() {
    if (this.targetX !== null && this.targetY !== null) {
      this.x += (this.targetX - this.x) * 0.05;
      this.y += (this.targetY - this.y) * 0.05;
    } else {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    }
  }
  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
  }
}

for (let i = 0; i < numParticles; i++) {
  particles.push(new Particle(Math.random() * canvas.width, Math.random() * canvas.height, i % 2 === 0 ? 'red' : 'white'));
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => {
    p.update();
    p.draw();
  });
  requestAnimationFrame(render);
}

function formTextShape(text, fontSize, color) {
  const textCanvas = document.createElement('canvas');
  const textCtx = textCanvas.getContext('2d');
  textCanvas.width = canvas.width;
  textCanvas.height = canvas.height;
  textCtx.fillStyle = color;
  textCtx.font = `bold ${fontSize}px Arial`;
  textCtx.fillText(text, canvas.width / 3, canvas.height / 2);

  const imageData = textCtx.getImageData(0, 0, canvas.width, canvas.height);
  const positions = [];
  for (let y = 0; y < canvas.height; y += 8) {
    for (let x = 0; x < canvas.width; x += 8) {
      const index = (y * canvas.width + x) * 4;
      if (imageData.data[index + 3] > 128) positions.push({ x, y });
    }
  }
  positions.sort(() => Math.random() - 0.5);
  particles.forEach((p, i) => {
    if (positions[i]) {
      p.targetX = positions[i].x;
      p.targetY = positions[i].y;
    }
  });
}

function cycleAnimation() {
  formTextShape('LexiGuardAI', 100, 'white');
  setTimeout(() => {
    particles.forEach(p => { p.targetX = null; p.targetY = null; });
    setTimeout(cycleAnimation, 4000);
  }, 4000);
}

render();
setTimeout(cycleAnimation, 2000);
</script>
</body>
</html>
"""

st.components.v1.html(html_code, height=600)

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
