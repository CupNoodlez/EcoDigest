import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TextClassificationPipeline, 
    AutoModelForSeq2SeqLM
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Sentiment Analysis Model ---
class SentimentAnalyzer:
    """Wrapper for the fine-tuned RoBERTa Sentiment Model."""
    def __init__(self, model_dir: str = "Model_ROBERTA-Sentiment"):
        self.model_path = Path(model_dir)
        if not self.model_path.exists():
             # Fallback: check relative to current working directory if generic path provided
             self.model_path = Path(os.getcwd()) / model_dir
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Sentiment Model not found at {self.model_path}")

        self.device = 0 if torch.cuda.is_available() else -1
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             self.device = "mps"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            
            # MPS handling
            if self.device == "mps":
                self.model.to("mps")
                
            pipeline_device = 0 if isinstance(self.device, int) and self.device >= 0 else -1
            
            self.pipe = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True,
                device=pipeline_device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Sentiment Model: {e}")

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Returns list of dicts: {'label': 'Positive'/'Negative', 'score': float, 'text': str}
        """
        results = []
        # Batch processing is handled by pipeline, but we iterate to format output
        raw_outputs = self.pipe(texts)

        for text, output in zip(texts, raw_outputs):
            scores = {item['label']: item['score'] for item in output}
            predicted_label = max(scores, key=scores.get)
            
            # Simple tie-break logic or preference could go here
            results.append({
                "text": text,
                "label": predicted_label,
                "confidence": scores[predicted_label],
                "scores": scores
            })
        return results

# --- 2. Centroid Ranking (Representative Comments) ---
class CentroidRanker:
    """Wrapper for SentenceTransformer Centroid Ranking."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # This downloads/loads from cache automatically
        self.model = SentenceTransformer(model_name)

    def get_representative_comments(self, comments: List[str], k: int = 10) -> List[str]:
        """Selects top k comments closest to the centroid."""
        if not comments:
            return []
        
        if len(comments) <= k:
            return comments

        embeddings = self.model.encode(comments)
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, centroid).flatten()
        
        # Argsort gives indices of sorted array (low to high), so reverse for high to low
        top_indices = similarities.argsort()[::-1][:k]
        return [comments[i] for i in top_indices]

# --- 3. Summarization Model ---
class Summarizer:
    """Wrapper for FLAN-T5 fine-tuned Summarizer."""
    def __init__(self, model_dir: str = "Model_FLAN-T5"):
        self.model_path = Path(model_dir)
        if not self.model_path.exists():
            self.model_path = Path(os.getcwd()) / model_dir
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"Summarizer Model not found at {self.model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # MPS support for T5 generation can be tricky, sticking to cpu/cuda for safety purely for now unless explicit
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             self.device = "mps"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path)).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Summarizer Model: {e}")

    def summarize(self, text: str) -> str:
        """Generates a summary for the given text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                do_sample=False
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
