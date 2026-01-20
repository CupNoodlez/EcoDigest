import pandas as pd
import re

# 1. Load the dataset
filename = 'datasets/41k-50k.csv'
df = pd.read_csv(filename)

# 2. Define the Cleaning Function
def clean_text_final(text):
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove RT prefix
    text = re.sub(r'^RT\s+:?', '', text, flags=re.IGNORECASE)
    
    # Remove Mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Clean Artifacts
    text = text.replace('$q$', "'").replace('â€™', "'").replace('Ã¢â‚¬Â¦', '...')
    
    # Remove leading non-alphanumeric chars
    text = re.sub(r'^[\s:.-]+', '', text)
    
    # Trim whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Apply Cleaning
# Identify correct column name (likely 'cleaned_message' from previous steps)
input_col = 'cleaned_message' if 'cleaned_message' in df.columns else 'message'
df['cleaned_message'] = df[input_col].apply(clean_text_final)

# 4. Semantic Filtering
# REMOVE only Sentiment 2 (News)
# We KEEP Sentiment 1, 0, -1 (even if political)
df_filtered = df[df['sentiment'] != 2].copy()

# 5. Deduplication
df_filtered = df_filtered.drop_duplicates(subset=['cleaned_message'])

# 6. Remove empty rows
df_filtered = df_filtered[df_filtered['cleaned_message'] != '']

# 7. Save to the SAME CSV file (Overwrite)
df_filtered.to_csv(filename, index=False)

print(f"Success! Overwrote '{filename}' with cleaned data (Political content KEPT).")
print(f"Final Count: {len(df_filtered)}")
