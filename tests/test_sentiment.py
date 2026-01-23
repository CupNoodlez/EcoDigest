import os
import sys
import torch
from pathlib import Path
from typing import Union, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

class ClimateSentimentClassifier:
    """
    A portable wrapper for the fine-tuned Environmental Sentiment RoBERTa model.
    Directory dependent: Expects 'Model_ROBERTA-Sentiment' to be in the parent directory of this script.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        # 1. DYNAMIC PATHING
        # If no path provided, look for 'Model_ROBERTA-Sentiment' in the parent folder of this script.
        if model_dir is None:
            current_script_path = Path(__file__).resolve().parent
            self.model_path = current_script_path.parent / "Model_ROBERTA-Sentiment"
        else:
            self.model_path = Path(model_dir)

        # 2. VALIDATION
        # Check if the folder exists and has the critical weights file
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Model directory not found at: {self.model_path}")
        
        has_weights = (self.model_path / "model.safetensors").exists() or (self.model_path / "pytorch_model.bin").exists()
        if not has_weights:
            raise FileNotFoundError(f"❌ Model weights (model.safetensors) missing in: {self.model_path}")

        # 3. DEVICE DETECTION (Includes Mac M1/M2/M3 Support)
        self.device = -1 # Default to CPU
        if torch.cuda.is_available():
            self.device = 0 # NVIDIA GPU
            print(f"🚀 ClimateClassifier loaded on: CUDA (GPU)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             self.device = "mps" # Apple Silicon
             print(f"🍏 ClimateClassifier loaded on: Apple MPS (Metal)")
        else:
            print(f"🐢 ClimateClassifier loaded on: CPU")

        # 4. LOAD MODEL & TOKENIZER
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            
            # Move model to Apple MPS manually if needed (Pipeline handles CUDA/CPU automatically)
            if self.device == "mps":
                self.model.to("mps")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}. Error: {e}")

        # 5. INITIALIZE PIPELINE
        # Note: We pass device=-1 for MPS because HF Pipelines have mixed support for MPS device IDs. 
        # We manually moved the model to MPS above, so inference will still be fast.
        pipeline_device = 0 if isinstance(self.device, int) and self.device >= 0 else -1
        
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=pipeline_device
        )

    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Accepts a string or list of strings.
        Returns a dictionary with: label, confidence, and sentiment_score (-1 to 1).
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        results = []
        
        # Run inference
        raw_outputs = self.pipe(text)

        for output in raw_outputs:
            # Output format is [[{'label': 'Negative', 'score': 0.1}, ...]]
            
            # Convert list of dicts to a single dict: {'Negative': 0.1, 'Neutral': 0.8, ...}
            scores = {item['label']: item['score'] for item in output}
            
            # Determine winner
            predicted_label = max(scores, key=scores.get)
            confidence = scores[predicted_label]
            
            # Calculate a normalized "sentiment score" from -1 (Neg) to +1 (Pos)
            # Formula: Positive Score - Negative Score
            neg_score = scores.get('Negative', 0.0)
            pos_score = scores.get('Positive', 0.0)
            normalized_score = pos_score - neg_score

            results.append({
                "label": predicted_label,
                "confidence": round(confidence, 4),
                "sentiment_score": round(normalized_score, 4),
                "details": scores
            })

        return results[0] if is_single else results

def main():
    print("⏳ Loading Climate Model... (this may take a moment)")
    
    try:
        # Initialize the model
        classifier = ClimateSentimentClassifier()
        print("\n" + "="*50)
        print("🌍 CLIMATE SENTIMENT MODEL IS READY")
        print("="*50)
        print("Type a sentence to analyze sentiment.")
        print("Type 'quit' or 'exit' to stop.")
        print("-" * 50)

        while True:
            # Get user input
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not text:
                continue

            # Run Prediction
            result = classifier.predict(text)
            
            # Print Result nicely
            label = result['label']
            conf = result['confidence']
            score = result['sentiment_score']
            
            # Visual formatting
            icon = "😐"
            if label == "Positive": icon = "✅"
            if label == "Negative": icon = "❌"
            
            print(f"   {icon} Sentiment:  {label}")
            print(f"   📊 Confidence: {conf:.1%}")
            print(f"   🧮 Score:      {score:.4f}")

    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()