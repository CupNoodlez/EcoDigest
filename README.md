# Summarized-Sentiment-Analyzer

## Notebooks

This project contains three main notebooks:

1. **`Model_Training.ipynb`** - Fine-tune RoBERTa model for sentiment classification (3-class: Negative/Neutral/Positive)
   - Complete training pipeline from data loading to model evaluation
   - Saves trained model to `models/roberta-environmental-sentiment-best/`
   - Generates `predictions_with_metadata.csv` for downstream analysis
   - ~3 hours training time on 10K samples (CPU)

2. **`Sentiment_Analysis_Streamlined.ipynb`** - Quick sentiment analysis using trained model
   - **Quick Mode**: Load pre-computed predictions instantly
   - **Inference Mode**: Apply fine-tuned model to new data
   - Interactive Plotly visualizations (temporal trends, platform comparison, geographic analysis)
   - No training required

3. **`Advanced_Sentiment_Analytics.ipynb`** - Comprehensive text analytics
   - Hashtag frequency analysis by sentiment
   - Word clouds with sentiment-specific color schemes
   - Named Entity Recognition (people, organizations, locations)
   - Text preprocessing and lemmatization
   - Keyword extraction using CountVectorizer
   - Uses predictions from trained model (no training step)

## Quick Start

**Option 1: Train from scratch**
```bash
# Run Model_Training.ipynb to train the model (takes ~3 hours)
```

**Option 2: Use pre-trained model**
```bash
# If you already have predictions_with_metadata.csv:
# - Run Sentiment_Analysis_Streamlined.ipynb for quick visualizations
# - Run Advanced_Sentiment_Analytics.ipynb for deep text analysis
```

## Output Files

- `models/roberta-environmental-sentiment-best/` - Fine-tuned RoBERTa model
- `predictions_with_metadata.csv` - Test set predictions with confidence scores and metadata (1,323 rows)