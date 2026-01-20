import pandas as pd
import re

# 1. Load the dataset
filename = 'datasets/41k-50k.csv'
df = pd.read_csv(filename)

# 2. Define the Advanced Cleaning Function
def clean_text_advanced(text):
    text = str(text)
    
    # Noise Removal
    text = re.sub(r'http\S+', '', text) # URLs
    text = re.sub(r'^RT\s+:?', '', text, flags=re.IGNORECASE) # RTs
    text = re.sub(r'@\w+', '', text) # Mentions
    
    # Artifacts
    text = text.replace('$q$', "'")
    text = text.replace('Ã¢â‚¬Â¦', '...')
    text = text.replace('â€™', "'")
    
    # Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Apply Cleaning
df['message'] = df['message'].apply(clean_text_advanced)

# 4. Deduplication
df = df.drop_duplicates(subset=['message'])

# 5. Semantic Filtering (Remove '2' = News)
df = df[df['sentiment'] != 2]

# 6. Save to the SAME CSV file (Overwrite)
# Keeping only relevant columns
df[['sentiment', 'message']].to_csv(filename, index=False)

print(f"Success! Overwrote '{filename}' with cleaned data.")
