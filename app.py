import streamlit as st
import yt_dlp
import pandas as pd
import requests
import os
import json
import plotly.express as px
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
# Make sure your .env file has: OPENROUTER_API_KEY=your_key_here
api_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="AI Social Intelligence", layout="wide")
st.title("🎯 YouTube Audience Intelligence Hub")

# Helper Function for OpenRouter API Calls
def query_openrouter(prompt, model_id="google/gemini-2.0-flash-001", is_json=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8501", 
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            st.error(f"OpenRouter Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# Initialize Session States
if 'csv_ready' not in st.session_state:
    st.session_state.csv_ready = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'comments' not in st.session_state:
    st.session_state.comments = []
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'last_ai_response' not in st.session_state:
    st.session_state.last_ai_response = None

# Verify API Key
if not api_key:
    st.error("❌ OpenRouter API Key not found! Check your .env file.")
    st.stop()

# --- STEP 1: INPUT & SCRAPE ---
st.header("Video Comments")
url = st.text_input("Paste YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Start Scraping"):
    if url:
        with st.spinner("🕵️ Scraping..."):
            try:
                ydl_opts = {'getcomments': True, 'max_comments': 100, 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    comments = [c['text'] for c in info.get('comments', [])]
                    if comments:
                        st.session_state.comments = comments
                        st.session_state.csv_ready = True
                        st.session_state.chart_data = None  # Reset chart for new video
                        st.success(f"✅ Found {len(comments)} comments!")
                    else:
                        st.warning("No comments found or comments are disabled for this video.")
            except Exception as e:
                st.error(f"Scraping Error: {e}")

# --- STEP 2: LOAD DATA ---
if st.session_state.csv_ready and not st.session_state.data_loaded:
    st.divider()
    if st.button("📥 Load Data into Intelligence System"):
        st.session_state.data_loaded = True
        st.rerun()

# --- STEP 3: INTERACTIVE Q&A + CHART ---
if st.session_state.data_loaded:
    st.divider()
    
    # 1. Automatic Sentiment Extraction
    if st.session_state.chart_data is None:
        with st.spinner("Generating Chart..."):
            chart_prompt = (
                f"Analyze these comments: {st.session_state.comments[:50]}. "
                "Return ONLY a JSON object: {'positive_score': number, 'negative_score': number}. "
                "The numbers should represent the percentage and add up to 100."
            )
            raw_response = query_openrouter(chart_prompt, is_json=True)
            
            if raw_response:
                try:
                    clean_json = raw_response.strip().replace('```json', '').replace('```', '')
                    st.session_state.chart_data = json.loads(clean_json)
                except Exception as e:
                    st.error("AI was unable to format the chart data properly.")

    # 2. Display the Pie Chart
    if st.session_state.chart_data:
        st.header("Audience Sentiment Pulse")
        s_data = st.session_state.chart_data
        pos = s_data.get('positive_score', 0)
        neg = s_data.get('negative_score', 0)
        
        pie_df = pd.DataFrame({
            "Sentiment": ["Positive ✅", "Negative ❌"],
            "Percentage": [pos, neg]
        })
        
        fig = px.pie(
            pie_df, 
            values='Percentage', 
            names='Sentiment',
            hole=0.5, 
            color='Sentiment',
            color_discrete_map={'Positive ✅': '#2ecc71', 'Negative ❌': '#e74c3c'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Analysis shows {pos}% Positive and {neg}% Negative feedback.")

    # --- STEP 4: DEEP-DIVE Q&A ---
    st.divider()
    st.subheader("💬 Deep-Dive Q&A")
    
    col1, col2, col3 = st.columns(3)
    preset_query = None
    with col1:
        if st.button("💰 Price related"): 
            preset_query = "List comments about price/value. Format ID_X: Comment"
    with col2:
        if st.button("😊 positive comments"): 
            preset_query = "List positive comments. Format ID_X: Comment"
    with col3:
        if st.button("😡 hate comments"): 
            preset_query = "List aggressive/hate comments. Format ID_X: Comment"

    custom_q = st.text_input("Ask a question about the feedback:", placeholder="e.g. Give me 5 negative comments...")
    final_query = custom_q if custom_q else preset_query

    if final_query:
        numbered_context = "\n".join([f"ID_{i+1}: {c}" for i, c in enumerate(st.session_state.comments[:100])])
        full_prompt = f"Format strictly as ID_X: Comment. Context:\n{numbered_context}\n\nQuestion: {final_query}"
        
        with st.spinner("AI is thinking..."):
            ans = query_openrouter(full_prompt)
            if ans:
                st.session_state.last_ai_response = ans 
                st.markdown("### 🤖 AI Insight")
                
                lines = [line.strip() for line in ans.split('\n') if line.strip()]
                for line in lines:
                    if "positive" in final_query.lower():
                        st.success(line)
                    elif "hate" in final_query.lower():
                        st.error(line)
                    else:
                        st.info(line)

    # --- RESET BUTTON ---
    st.divider()
    if st.button("🗑️ Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()