import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
import os

from pathlib import Path

def run_summarization():
    current_script_path = Path(__file__).resolve().parent
    model_path = current_script_path.parent / "Model_FLAN-T5"
    data_path = current_script_path.parent / "datasets" / "dataset_test.csv"
    
    if not model_path.exists():
        print(f"Error: Model path not found at {model_path}")
        return
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    num_examples = 2
    examples = df.head(num_examples)

    print("\n" + "="*60)
    print(f"GENERATING SUMMARIES FOR {num_examples} EXAMPLES")
    print("="*60)

    for index, row in examples.iterrows():
        input_text = row['input_text']
        target_text = row.get('target_text', 'N/A')

        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                do_sample=False
            )

        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nExample {index + 1}:")
        print("-" * 30)
        print(f"INPUT:\n{input_text}\n")
        print(f"TARGET SUMMARY:\n{target_text}\n")
        print(f"GENERATED SUMMARY:\n{generated_summary}")
        print("-" * 30)

if __name__ == "__main__":
    run_summarization()