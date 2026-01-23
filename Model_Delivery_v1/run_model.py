import os
import sys

# Ensure we can find the package
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from climate_sentiment_pkg.classifier import ClimateSentimentClassifier
except ImportError:
    print("❌ Error: Could not find 'climate_sentiment_pkg'.")
    print("Make sure this script is next to the 'climate_sentiment_pkg' folder.")
    input("Press Enter to exit...")
    sys.exit(1)

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