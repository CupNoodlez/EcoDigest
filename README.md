# 🧠 Sentiment & Consensus Analyzer

A multi-stage NLP pipeline designed to analyze environmental sentiment and extract representative consensus from public comments. The project leverages fine-tuned transformer models for classification and summarization, combined with semantic clustering for representative insight extraction.

---

## 🚀 Applications

### 1. Unified Sentiment & Consensus Analyzer
**File:** `sentiment_summarizer_app.py`  
The primary interactive tool. Input a list of raw comments and the AI will:
1.  **Sentiment Analysis**: Classify each comment as Positive or Negative using a fine-tuned **RoBERTa** model.
2.  **Centroid Ranking**: Identified the most representative "centroid" comments for each group using **MiniLM** embeddings.
3.  **Consensus Summarization**: Generate a concise summary of the group's collective opinion using a fine-tuned **FLAN-T5** model.

**To Run:**
```bash
streamlit run sentiment_summarizer_app.py
```

### 2. Public Perception Dashboard
**File:** `dashboard_app.py`  
A visualization suite for large-scale batch analysis (requires `model_output.csv`).
- Sentiment distribution and over-time trends.
- Interactive Word Clouds and composición analysis.
- Raw data explorer.

**To Run:**
```bash
streamlit run dashboard_app.py
```

---

## 🛠 Project Structure

- `Model_ROBERTA-Sentiment/`: Fine-tuned model for environmental sentiment classification.
- `Model_FLAN-T5/`: Fine-tuned model for high-fidelity summarization.
- `utils.py`: Centralized logic for model loading and inference pipelines.
- `datasets/`: Training and testing data, including the `summarization_dataset.csv` for sample testing.
- `tests/`: Standalone CLI scripts for testing individual model performance (`test_sentiment.py`, `test_summ.py`).
- `requirements.txt`: Full dependency list.

---

## 🧬 Scientific Pipeline

1.  **Sentiment Classification**: Based on `cardiffnlp/twitter-roberta-base-sentiment`, fine-tuned on climate-specific datasets.
2.  **Ranking (Centroid)**: Uses `all-MiniLM-L6-v2` to vectorize comments. We calculate the mean vector (centroid) of each cluster and select the top $k$ comments with the highest cosine similarity to the centroid.
3.  **Summarization**: A sequence-to-sequence model (FLAN-T5) fine-tuned with specific prompts:
    - *Positive:* "Summarize the opinions of users who believe in the reality of climate change.:"
    - *Negative:* "Summarize the opinions of users who skeptical of/deny climate change.:"

---

## 📋 Installation

1. Create and activate a environment (Conda or venv):
```bash
conda create -n sentiment-analysis python=3.12
conda activate sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure models are placed in the root directory (refer to the **Model Files** section in structure).

---

## 📚 Original Research & Training
The core models were developed and analyzed in the following notebooks:
- `Model_Training.ipynb`: Original RoBERTa fine-tuning process.
- `FLANT5-FT.ipynb`: Dataset preparation and fine-tuning for the summarization model.
- `Advanced_Sentiment_Analytics.ipynb`: Initial exploration of keyword extraction and NER.