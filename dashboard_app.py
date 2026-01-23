import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from climate_sentiment_pkg.classifier import ClimateSentimentClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Public Perception Dashboard",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def get_classifier():
    # This loads the model once and keeps it in memory
    return ClimateSentimentClassifier()

# Load the model quietly in the background
classifier = get_classifier()

# 2. Load Data
@st.cache_data
def load_data():
    try:
        # Load the file
        df = pd.read_csv("model_output.csv") # OR "model_output.csv"
        
        # --- AUTO-DETECT TEXT COLUMN ---
        # We look for common names and rename the first match to 'cleaned_message'
        possible_names = ['cleaned_message', 'message', 'text', 'tweet', 'content']
        found_col = None
        for col in possible_names:
            if col in df.columns:
                found_col = col
                break
        
        if found_col:
            df = df.rename(columns={found_col: 'cleaned_message'})
        else:
            st.error(f"❌ Error: Could not find a text column. Your file columns are: {list(df.columns)}")
            st.stop()
            
        # Date Logic
        if 'date' not in df.columns and 'tweetid' in df.columns:
            df['timestamp'] = ((df['tweetid'] >> 22) + 1288834974657) / 1000
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            
        return df

    except FileNotFoundError:
        st.error("File not found. Please make sure 'model_output.csv' exists.")
        st.stop()

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Filter Options")

# Filter by Sentiment
if 'predicted_sentiment' in df.columns:
    sentiment_filter = st.sidebar.multiselect(
        "Select Sentiment:",
        options=df['predicted_sentiment'].unique(),
        default=df['predicted_sentiment'].unique()
    )
    # Apply Sentiment Filter
    df_filtered = df[df['predicted_sentiment'].isin(sentiment_filter)]
else:
    st.warning("Column 'predicted_sentiment' missing. Showing all data.")
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
    col2.metric("Negative Sentiment", f"{(neg_count/total_tweets)*100:.1f}%", delta_color="inverse")
    col3.metric("Positive Sentiment", f"{(pos_count/total_tweets)*100:.1f}%")
else:
    st.warning("No data available with current filters.")

st.markdown("---")

# 5. Visualizations
c1, c2 = st.columns((2, 1))

with c1:
    if 'predicted_sentiment' in df_filtered.columns and not df_filtered.empty:
        st.subheader("Sentiment Distribution")
        fig_bar = px.bar(
            df_filtered['predicted_sentiment'].value_counts().reset_index(),
            x='predicted_sentiment', 
            y='count',
            color='predicted_sentiment',
            color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
            title="Count of Opinions by Sentiment"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if 'date' in df.columns:
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
        st.subheader("Sentiment Composition")
        fig_pie = px.pie(
            df_filtered, 
            names='predicted_sentiment',
            color='predicted_sentiment',
            color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Word Cloud")
    if st.button("Generate Word Cloud"):
        if not df_filtered.empty:
            # Combine all text
            text = " ".join(str(review) for review in df_filtered.cleaned_message.dropna())
            
            if len(text.strip()) > 0:
                # Create Cloud
                wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Not enough text to generate a word cloud.")
        else:
            st.warning("No data selected.")

# 6. Raw Data Explorer
st.markdown("---")
st.subheader("Raw Data Explorer")

# Display whatever columns we actually have
cols_to_show = [c for c in ['predicted_sentiment', 'confidence', 'cleaned_message'] if c in df_filtered.columns]
st.dataframe(df_filtered[cols_to_show].head(100))

# --- NEW: SIDEBAR FOR LIVE ANALYSIS ---
with st.sidebar:
    st.header("🤖 Live Model Test")
    st.write("Type a sentence to test the model in real-time:")
    
    user_input_text = st.text_area("Enter text here:", height=100)
    
    if st.button("Analyze Text"):
        if user_input_text.strip():
            with st.spinner("Analyzing..."):
                result = classifier.predict(user_input_text)
            
            # Display Result
            st.success(f"Sentiment: **{result['label']}**")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Show detailed scores if available
            if 'details' in result:
                st.json(result['details'])
        else:
            st.warning("Please enter some text first.")