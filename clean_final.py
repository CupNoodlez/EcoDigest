import pandas as pd
import re

# 1. Load the dataset (The one you just shared)
filename = 'datasets/41k-50k.csv'
df = pd.read_csv(filename)

# 2. Define the Advanced Cleaning Function
def clean_text_final(text):
    text = str(text)
    
    # Remove ALL @mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove "RT" artifacts if any exist
    text = re.sub(r'^RT\s+:?', '', text, flags=re.IGNORECASE)
    
    # Remove URLs (just in case any are left)
    text = re.sub(r'http\S+', '', text)
    
    # Fix artifacts (like the $q$ or smart quotes if they remain)
    text = text.replace('$q$', "'").replace('â€™', "'").replace('Ã¢â‚¬Â¦', '...')
    
    # Remove non-alphanumeric characters at the start (e.g., ": " or "- ") often left after removing RTs
    text = re.sub(r'^[\s:.-]+', '', text)

    # Trim extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Apply Cleaning
# We operate on the 'cleaned_message' column since that's what exists in this file
df['cleaned_message'] = df['cleaned_message'].apply(clean_text_final)

# 4. Semantic Filtering
# REMOVE Sentiment 2 (News/Objective)
# Keep only 1 (Pro), 0 (Neutral), -1 (Anti)
df_filtered = df[df['sentiment'] != 2].copy()

# 5. Deduplication
# Remove duplicate messages
df_filtered = df_filtered.drop_duplicates(subset=['cleaned_message'])

# 6. Final Formatting & Save
# Overwrite the file with the clean version
df_filtered.to_csv(filename, index=False)

print(f"Success! '{filename}' has been fully cleaned.")
print(f"Original Count: {len(df)}")
print(f"Final Count: {len(df_filtered)}")
print(f"Removed: {len(df) - len(df_filtered)} rows (News & Duplicates)")
print("\nSample of Cleaned Data:")
print(df_filtered.head())
