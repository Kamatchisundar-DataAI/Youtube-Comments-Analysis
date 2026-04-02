import streamlit as st
import yt_dlp
import pandas as pd
import requests
import os
import json
import plotly.express as px
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION ---
if "OPENROUTER_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="AI Social Intelligence", layout="wide")
st.title("🎯 YouTube Audience Intelligence Hub")

# --- 2. CACHED SCRAPING FUNCTION ---
@st.cache_data(show_spinner=False)
def scrape_youtube_comments(url):
    try:
        ydl_opts = {'getcomments': True, 'max_comments': 100, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            comments = [c['text'] for c in info.get('comments', [])]
            return comments
    except Exception as e:
        return f"Error: {e}"

# Helper Function for OpenRouter API Calls
def query_openrouter(prompt, model_id="google/gemini-2.0-flash-001", is_json=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://youtube-comments-analysis-sentiment-analyze.streamlit.app/", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}]
    }
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except Exception:
        return None

# Initialize Session States
if 'csv_ready' not in st.session_state: st.session_state.csv_ready = False
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'comments' not in st.session_state: st.session_state.comments = []
if 'chart_data' not in st.session_state: st.session_state.chart_data = None

# --- STEP 1: INPUT & SCRAPE ---
st.header("Video Comments")
url = st.text_input("Paste YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
st.caption("💡 **Example URL:** `https://www.youtube.com/watch?v=dQw4w9WgXcQ` (Paste this to test!)")

if st.button("🚀 Start Scraping"):
    if url:
        with st.spinner("🕵️ Scraping..."):
            result = scrape_youtube_comments(url)
            if isinstance(result, list) and len(result) > 0:
                st.session_state.comments = result
                st.session_state.csv_ready = True
                st.session_state.chart_data = None 
                st.success(f"✅ Found {len(result)} comments!")
            elif "Error" in str(result):
                st.error(result)
            else:
                st.warning("No comments found. They might be disabled.")

# --- STEP 2: LOAD DATA ---
if st.session_state.csv_ready and not st.session_state.data_loaded:
    st.divider()
    if st.button("📥 Load Data into Intelligence System"):
        st.session_state.data_loaded = True
        st.rerun()

# --- STEP 3: ANALYZE & CHART ---
if st.session_state.data_loaded:
    st.divider()
    
    if st.session_state.chart_data is None:
        with st.spinner("Generating Chart..."):
            # FIXED: Removed 'genai.GenerativeModel' and used 'query_openrouter' instead
            chart_prompt = (
                f"Analyze these comments: {st.session_state.comments[:50]}. "
                "Return ONLY a JSON object: {'positive_score': number, 'negative_score': number}. "
                "The numbers must be whole integers that add up to 100."
            )
            raw_response = query_openrouter(chart_prompt, is_json=True)
            
            if raw_response:
                try:
                    # Clean response and load JSON
                    clean_json = raw_response.strip().replace('```json', '').replace('```', '')
                    st.session_state.chart_data = json.loads(clean_json)
                except Exception:
                    st.error("AI was unable to format the chart data properly.")

    # Display Chart
    if st.session_state.chart_data:
        st.header("Audience Sentiment Pulse")
        s_data = st.session_state.chart_data
        pos = s_data.get('positive_score', 0)
        neg = s_data.get('negative_score', 0)
        
        pie_df = pd.DataFrame({
            "Sentiment": ["Positive ✅", "Negative ❌"],
            "AI_Score": [pos, neg]
        })
        
        fig = px.pie(
            pie_df, 
            values='AI_Score', 
            names='Sentiment', 
            hole=0.5,
            color='Sentiment', 
            color_discrete_map={'Positive ✅': '#2ecc71', 'Negative ❌': '#e74c3c'},
            hover_data=['AI_Score'],
            labels={'AI_Score': 'Percentage'}
        )

        fig.update_traces(
            textinfo='percent+label', 
            hovertemplate="%{label}: %{value}%<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Analysis shows {pos}% Positive and {neg}% Negative feedback.")

    st.divider()
    st.subheader("💬 Deep-Dive Q&A about comments")
    
    col1, col2, col3 = st.columns([1,1,1])
    preset_query = None
    with col1:
        if st.button("💰 Price related"): preset_query = "List price/value comments. Format ID_X: Comment"
    with col2:
        if st.button("😊 Positive comments"): preset_query = "List positive comments. Format ID_X: Comment"
    with col3:
        if st.button("😡 Hate comments"): preset_query = "List hate comments. Format ID_X: Comment"

    custom_q = st.text_input("Ask a question:", placeholder="e.g. Most common complaint?")
    final_query = custom_q if custom_q else preset_query

    if final_query:
        numbered_context = "\n".join([f"ID_{i+1}: {c}" for i, c in enumerate(st.session_state.comments[:100])])
        full_prompt = f"Format strictly ID_X: Comment. Context:\n{numbered_context}\n\nQuestion: {final_query}"
        
        with st.spinner("AI is thinking..."):
            ans = query_openrouter(full_prompt)
            if ans:
                st.markdown("### 🤖 AI Insight")
                lines = [line.strip() for line in ans.split('\n') if line.strip()]
                for line in lines:
                    if "positive" in final_query.lower(): st.success(line)
                    elif "hate" in final_query.lower(): st.error(line)
                    else: st.info(line)

    if st.button("🗑️ Reset"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
