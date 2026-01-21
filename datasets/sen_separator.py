import pandas as pd
import sys
import os

def split_sentiment_csv(input_filename):
    """
    Reads a CSV, separates rows by sentiment (-1 and 1), ignoring 0,
    and saves them to two separate CSV files.
    """
    
    # Check if file exists
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(input_filename)
        
        # Ensure the 'sentiment' column exists to avoid errors
        if 'sentiment' not in df.columns:
            print("Error: The CSV does not contain a 'sentiment' column.")
            return

        # Filter for negative sentiment (-1)
        negative_df = df[df['sentiment'] == -1]
        
        # Filter for positive sentiment (1)
        positive_df = df[df['sentiment'] == 1]
        
        # Define output filenames
        neg_output = 'negative_sentiment.csv'
        pos_output = 'positive_sentiment.csv'
        
        # Save the separated data to new CSV files
        # index=False ensures we don't save the row numbers as a column
        negative_df.to_csv(neg_output, index=False)
        positive_df.to_csv(pos_output, index=False)
        
        print(f"Successfully processed '{input_filename}'.")
        print(f" - Saved {len(negative_df)} negative rows to '{neg_output}'")
        print(f" - Saved {len(positive_df)} positive rows to '{pos_output}'")
        print(f" - Ignored {len(df) - len(negative_df) - len(positive_df)} rows (neutral/other)")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_file = input("Enter the input CSV filename: ")
    split_sentiment_csv(input_file)
