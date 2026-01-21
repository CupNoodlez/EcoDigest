import pandas as pd
import re
import sys
from langdetect import detect, LangDetectException

def clean_strict_final(input_path, output_path):
    try:
        # Load dataset
        # header=None assumes no column names. 
        # Col 0 = Sentiment, Col 1 = Text, Col 2 = ID
        df = pd.read_csv(input_path, header=None)
        
        initial_count = len(df)
        print(f"Initial rows: {initial_count}")

        # --- STEP 0: SENTIMENT FILTER ---
        # Remove rows where Sentiment Score (Column 0) is 2
        # We convert to string first to handle both int(2) and str("2") safely
        df = df[df[0].astype(str) != '2']
        
        count_after_sentiment = len(df)
        print(f"Removed {initial_count - count_after_sentiment} rows with Sentiment Score 2.")

        # Define the Text Cleaning Logic
        def fix_truncation(text):
            if not isinstance(text, str): return None
            text = text.strip()
            
            # --- PHASE 1: ARTIFACT REMOVAL ---
            text = text.replace(" 's ", "")
            text = re.sub(r'(?<=[.?!…])\s+[a-zA-Z]$', '', text)
            text = re.sub(r'(?<=\.\.\.)\s+[a-zA-Z]$', '', text) 

            # --- PHASE 2: STRICT CLEANING (Truncation) ---
            text = re.sub(r'([.?!])\s*\.\.\.$', r'\1', text)
            
            if text.endswith('...'):
                content_before_dots = text[:-3]
                match = re.search(r'(.*(?<!\.)[.?!](?!\.))', content_before_dots)
                if match:
                    text = match.group(1)
                else:
                    return None
            
            if text.endswith(':'):
                text = text[:-1]

            # --- PHASE 3: VALIDATION ---
            
            # Check 1: Length (> 30 chars)
            if len(text) < 30:
                return None
            
            # Check 2: Topic (Must contain keywords)
            text_lower = text.lower()
            if "climate" not in text_lower and "global warming" not in text_lower:
                return None

            # Check 3: Language Detection (Must be English)
            try:
                if detect(text) != 'en':
                    return None
            except LangDetectException:
                return None
                
            return text

        # Apply text cleaning logic
        print("Running text cleaning and language detection (this may take a moment)...")
        # We work on a copy to avoid SettingWithCopy warnings
        df = df.copy()
        df[1] = df[1].apply(fix_truncation)
        
        # Drop rows that returned None
        clean_df = df.dropna(subset=[1])
        
        # Final Stats
        final_count = len(clean_df)
        total_removed = initial_count - final_count
        
        print(f"------------------------------------------------")
        print(f"Total Removed: {total_removed} rows")
        print(f"Remaining Rows: {final_count}")
        print(f"Data Retention: {(final_count/initial_count)*100:.2f}%")
        print(f"------------------------------------------------")
        
        # Save
        clean_df.to_csv(output_path, index=False, header=False)
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_csv = input("Enter input CSV : ")
    if not input_csv.endswith('.csv'): input_csv += '.csv'
    output_csv = input_csv.replace('.csv', '_final_processed.csv')
    
    clean_strict_final(input_csv, output_csv)