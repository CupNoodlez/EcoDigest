import streamlit as st
import pandas as pd
from utils import SentimentAnalyzer, CentroidRanker, Summarizer

# --- Page Config ---
st.set_page_config(
    page_title="Sentiment & Consensus Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --- Load Models (Cached) ---
@st.cache_resource
def load_models():
    with st.spinner("Loading AI Models... (RoBERTa, FLAN-T5, MiniLM)"):
        sentiment_model = SentimentAnalyzer()
        centroid_model = CentroidRanker()
        summarizer_model = Summarizer()
    return sentiment_model, centroid_model, summarizer_model

try:
    sentiment_model, centroid_model, summarizer_model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# --- Main UI ---
st.title("🧠 Sentiment & Consensus Analyzer")
st.markdown("""
Input a list of comments below. The AI will:
1. **Analyze Sentiment** (Positive/Negative)
2. **Cluster & Rank** insights (Centroid Ranking)
3. **Summarize** the consensus for each group
""")

# Input Area
default_text = ""
import pandas as pd
import random

# Load sample data
@st.cache_data
def load_sample_comments():
    try:
        df = pd.read_csv("datasets/summarization_dataset.csv")
        samples = df['input_text'].sample(40).tolist()
        
        cleaned_samples = []
        for s in samples:
            lines = s.split('\n')
            comments_only = [line.strip().lstrip('- ') for line in lines if line.strip().startswith('-')]
            if comments_only:
                cleaned_samples.append("\n".join(comments_only))
        
        return cleaned_samples[:40]
    except Exception as e:
        return [f"Error loading inputs: {e}"]

sample_options = load_sample_comments()

with st.expander("📝 Try Sample Inputs", expanded=False):
    st.caption("Copy-paste these to test (40 random samples from dataset):")
    all_samples_text = "\n".join(sample_options)
    st.code(all_samples_text)

raw_text = st.text_area("Enter comments (one per line):", height=200, placeholder="Paste your comments here...")

if st.button("Analyze Comments", type="primary"):
    if not raw_text.strip():
        st.warning("Please enter some comments to analyze.")
    else:
        # Preprocessing
        comments = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        if not comments:
             st.warning("No valid comments found.")
        else:
            # 1. Sentiment Analysis
            with st.status("Running Analysis...", expanded=True) as status:
                st.write("🔍 Classifying Sentiment...")
                results = sentiment_model.predict(comments)
                
                # Group by label
                groups = {"Positive": [], "Negative": []}
                for res in results:
                    label = res['label']
                    if label in groups:
                        groups[label].append(res['text'])
                
                st.write(f"✅ Found {len(groups['Positive'])} Positive and {len(groups['Negative'])} Negative comments.")
                
                # 2. Centroid Ranking & 3. Summarization
                summaries = {}
                representative_comments = {}
                
                for label, group_comments in groups.items():
                    if not group_comments:
                        continue
                        
                    st.write(f"📊 Processing {label} group...")
                    
                    # Centroid Ranking
                    # If > 10, filter to top 10. If <= 10, use all.
                    # Note: CentroidRanker handles the "return all if <= k" logic internally too, 
                    # but explicit logic here helps with UI messaging if needed.
                    if len(group_comments) > 10:
                        top_comments = centroid_model.get_representative_comments(group_comments, k=10)
                        representative_comments[label] = top_comments
                    else:
                        representative_comments[label] = group_comments
                    
                    # Summarization
                    # Format as seen in FLANT5-FT.ipynb:
                    # "Summarize the opinions of users who [condition]: - comment 1 - comment 2 ..."
                    
                    if label == "Positive":
                        formatted_comments = "\n".join([f"- {c}" for c in representative_comments[label]])
                        prompt = f"Summarize the opinions of users who believe in the reality of climate change.:\n{formatted_comments}"
                    else:
                        formatted_comments = "\n".join([f"- {c}" for c in representative_comments[label]])
                        prompt = f"Summarize the opinions of users who skeptical of/deny climate change.:\n{formatted_comments}"

                    summary = summarizer_model.summarize(prompt)
                    summaries[label] = summary
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- Display Results ---
            st.divider()
            
            # Determine layout based on active groups
            active_labels = [label for label in ["Positive", "Negative"] if label in summaries]
            
            if not active_labels:
                st.info("No Positive or Negative comments identified (maybe they were Neutral?).")
            else:
                cols = st.columns(len(active_labels))
                
                for idx, label in enumerate(active_labels):
                    with cols[idx]:
                        color = "green" if label == "Positive" else "red"
                        icon = "✅" if label == "Positive" else "❌"
                        
                        st.subheader(f"{icon} {label} Consensus")
                        
                        # Summary Card
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 10px; background-color: rgba(128, 128, 128, 0.1); border-left: 5px solid {color};">
                            <strong>AI Summary:</strong><br>
                            {summaries[label]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("###") # Spacer
                        
                        # Representative Comments
                        st.markdown(f"**Key Representative Comments ({len(representative_comments[label])})**")
                        for comment in representative_comments[label]:
                            st.info(comment)
                        
                        # Show all comments in expander
                        with st.expander(f"See all {len(groups[label])} {label.lower()} comments"):
                            for c in groups[label]:
                                st.text(f"• {c}")

