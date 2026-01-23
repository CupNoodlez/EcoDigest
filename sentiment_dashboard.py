import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# 1. Page Configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Public Perception Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --- CRITICAL FIX: PATH SETUP ---
current_dir = Path(__file__).resolve().parent

# Add root dir to path to ensure we find the package
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# --- IMPORT MODEL ---
try:
    from climate_sentiment_pkg.classifier import ClimateSentimentClassifier
except ImportError:
    st.error(f"❌ Could not find 'climate_sentiment_pkg' package.")
    st.stop()

@st.cache_resource
def get_classifier():
    # Explicitly point to the local model folder
    model_path = Path(__file__).resolve().parent / "Model_ROBERTA-Sentiment"
    return ClimateSentimentClassifier(model_dir=str(model_path))

# Load the model quietly
try:
    classifier = get_classifier()
except Exception as e:
    st.error(f"❌ Model failed to load. Error: {e}")
    st.stop()


# 2. Load Data
@st.cache_data
def load_data():
    csv_file = "model_output.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        st.warning(f"⚠️ '{csv_file}' not found. Using placeholder data for demo.")
        return pd.DataFrame({'cleaned_message': [], 'predicted_sentiment': [], 'confidence': []})

    try:
        df = pd.read_csv(csv_file)
        
        # --- AUTO-DETECT TEXT COLUMN ---
        possible_names = ['cleaned_message', 'message', 'text', 'tweet', 'content']
        found_col = None
        for col in possible_names:
            if col in df.columns:
                found_col = col
                break
        
        if found_col:
            df = df.rename(columns={found_col: 'cleaned_message'})
        else:
            st.error(f"❌ Error: Could not find a text column. Available: {list(df.columns)}")
            st.stop()
            
        # --- ROBUST DATE LOGIC ---
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif 'tweetid' in df.columns:
            # Snowflake ID to Date conversion
            try:
                df['tweetid'] = pd.to_numeric(df['tweetid'], errors='coerce')
                df = df.dropna(subset=['tweetid'])
                df['timestamp'] = ((df['tweetid'].astype(int) >> 22) + 1288834974657) / 1000
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            except Exception:
                pass # Fail silently if IDs are bad
            
        return df

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

df = load_data()


# 3. Sidebar Filters
st.sidebar.header("Filter Options")

# Filter by Sentiment
if 'predicted_sentiment' in df.columns:
    all_sentiments = df['predicted_sentiment'].dropna().unique()
    sentiment_filter = st.sidebar.multiselect(
        "Select Sentiment:",
        options=all_sentiments,
        default=all_sentiments
    )
    df_filtered = df[df['predicted_sentiment'].isin(sentiment_filter)]
else:
    df_filtered = df

# Filter by Confidence Score
if 'confidence' in df.columns:
    min_confidence = st.sidebar.slider("Minimum Confidence Score:", 0.0, 1.0, 0.0)
    df_filtered = df_filtered[df_filtered['confidence'] >= min_confidence]


# 4. Main Dashboard UI
st.title("Environmental Policy Sentiment Dashboard")

# KPI Metrics
col1, col2, col3 = st.columns(3)
total_tweets = len(df_filtered)

if total_tweets > 0 and 'predicted_sentiment' in df.columns:
    neg_count = len(df_filtered[df_filtered['predicted_sentiment'] == 'Negative'])
    pos_count = len(df_filtered[df_filtered['predicted_sentiment'] == 'Positive'])
    
    col1.metric("Total Opinions", f"{total_tweets:,}")
    col2.metric("Negative Sentiment", f"{(neg_count/total_tweets)*100:.1f}%")
    col3.metric("Positive Sentiment", f"{(pos_count/total_tweets)*100:.1f}%")
else:
    st.info("No data available (check filters or CSV).")

st.markdown("---")

# 5. Visualizations
c1, c2 = st.columns((2, 1))

with c1:
    if 'predicted_sentiment' in df_filtered.columns and not df_filtered.empty:
        st.subheader("Sentiment Distribution")
        counts = df_filtered['predicted_sentiment'].value_counts().reset_index()
        counts.columns = ['predicted_sentiment', 'count']
        
        fig_bar = px.bar(
            counts,
            x='predicted_sentiment', 
            y='count',
            color='predicted_sentiment',
            color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
            title="Count of Opinions by Sentiment"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if 'date' in df.columns and not df_filtered['date'].isna().all():
            st.subheader("Sentiment Over Time")
            daily_counts = df_filtered.groupby([pd.Grouper(key='date', freq='D'), 'predicted_sentiment']).size().reset_index(name='count')
            fig_line = px.line(
                daily_counts, 
                x='date', 
                y='count', 
                color='predicted_sentiment',
                color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"}
            )
            st.plotly_chart(fig_line, use_container_width=True)

with c2:
    if 'predicted_sentiment' in df_filtered.columns and not df_filtered.empty:
        st.subheader("Composition")
        fig_pie = px.pie(
            df_filtered, 
            names='predicted_sentiment',
            color='predicted_sentiment',
            color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Word Cloud")
    if st.button("Generate Cloud"):
        if not df_filtered.empty and 'cleaned_message' in df_filtered.columns:
            text = " ".join(str(msg) for msg in df_filtered.cleaned_message.dropna())
            if len(text) > 10:
                wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Not enough text data.")
        else:
            st.warning("No data selected.")

# 6. Raw Data Explorer
st.markdown("---")
st.subheader("Raw Data Explorer")
if not df_filtered.empty:
    cols = [c for c in ['predicted_sentiment', 'confidence', 'cleaned_message'] if c in df_filtered.columns]
    st.dataframe(df_filtered[cols].head(50))

# --- LIVE MODEL SIDEBAR ---
with st.sidebar:
    st.markdown("---")
    st.header("🤖 Live Model Test")
    st.write("Type a sentence to test the model:")
    
    user_input = st.text_area("Enter text:", height=100)
    
    if st.button("Analyze"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                res = classifier.predict(user_input)
            
            # Result Display
            lbl = res['label']
            if lbl == "Positive":
                st.success(f"**{lbl}** ({res['confidence']:.1%})")
            elif lbl == "Negative":
                st.error(f"**{lbl}** ({res['confidence']:.1%})")
            else:
                st.info(f"**{lbl}** ({res['confidence']:.1%})")
                
            if 'details' in res:
                with st.expander("Show Scores"):
                    st.json(res['details'])
        else:
            st.warning("Please type something.")