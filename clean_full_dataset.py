import pandas as pd
import re

# 1. Load the FULL dataset
filename = 'datasets/twitter_sentiment_data.csv'
df = pd.read_csv(filename)

print(f"Initial Count: {len(df)}")

# 2. Define Cleaning Function
def clean_text_final(text):
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove RT prefix
    text = re.sub(r'^RT\s+:?', '', text, flags=re.IGNORECASE)
    
    # Remove Mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Clean Artifacts (including the ones seen in the head: Ã¢â‚¬Â¦)
    text = text.replace('$q$', "'").replace('â€™', "'").replace('Ã¢â‚¬Â¦', '...')
    text = text.replace('&amp;', '&') # Common HTML entity in large datasets
    
    # Remove leading non-alphanumeric chars
    text = re.sub(r'^[\s:.-]+', '', text)
    
    # Trim whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Apply Cleaning
df['cleaned_message'] = df['message'].apply(clean_text_final)

# 4. Filter: Remove News (Sentiment 2)
# Keeping 1 (Pro), 0 (Neutral), -1 (Anti)
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

# 7. Final Save
# We'll save just the sentiment and message, as tweetid might not be needed for training
df_final[['sentiment', 'cleaned_message']].to_csv('cleaned_twitter_data_final.csv', index=False)

print(f"Success! Final count: {len(df_final)}")
print(f"Removed: {len(df) - len(df_final)} rows (News, Duplicates, Truncated)")
print("\nSample of Cleaned Data:")
print(df_final[['sentiment', 'cleaned_message']].head())
