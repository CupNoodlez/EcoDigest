import pandas as pd
import re

# 1. Load the dataset
df = pd.read_csv('datasets/41k-50k.csv')

# 2. Define Cleaning Function
def clean_text_final(text):
    text = str(text)
    # Remove URLs, RTs, Mentions
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'^RT\s+:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', '', text)
    # Fix artifacts
    text = text.replace('$q$', "'").replace('â€™', "'").replace('Ã¢â‚¬Â¦', '...')
    # Clean whitespace
    text = re.sub(r'^[\s:.-]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Apply Cleaning (use existing column if available)
input_col = 'cleaned_message' if 'cleaned_message' in df.columns else 'message'
df['cleaned_message'] = df[input_col].apply(clean_text_final)

# 4. Filter: Remove News (Sentiment 2)
df_filtered = df[df['sentiment'] != 2].copy()

# 5. Filter: Remove Unfinished/Truncated Phrases
def is_unfinished(text):
    text = str(text).strip()
    # Remove if it ends with ellipsis (truncated)
    if text.endswith('...') or text.endswith('…'):
        return True
    # Remove if it's too short (< 3 words)
    if len(text.split()) < 3:
        return True
    return False

# Apply the unfinished filter
df_final = df_filtered[~df_filtered['cleaned_message'].apply(is_unfinished)].copy()

# 6. Deduplicate
df_final = df_final.drop_duplicates(subset=['cleaned_message'])

# 7. Save
df_final[['sentiment', 'cleaned_message']].to_csv('datasets/41k-50k.csv', index=False)

print(f"Success! Final count: {len(df_final)}")
