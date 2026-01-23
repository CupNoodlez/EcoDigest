# test_pkg.py
import sys
import os

# Optional: Ensure the current directory is in python path
sys.path.append(os.getcwd())

print("Attempting to import the package...")

try:
    # 1. Import your new wrapper
    from climate_sentiment_pkg.classifier import ClimateSentimentClassifier
    
    # 2. Initialize the model
    print("Initializing model (this might take a few seconds)...")
    model = ClimateSentimentClassifier()
    
    # 3. Define some test sentences
    test_sentences = [
        "The ice caps are melting and we need to act fast!",
        "Climate change is just a natural cycle, nothing to worry about.",
        "The weather is mild today."
    ]
    
    # 4. Run predictions
    print("\n" + "="*40)
    print("🧪 RUNNING TEST PREDICTIONS")
    print("="*40)
    
    for text in test_sentences:
        result = model.predict(text)
        
        # Unpack result for clean printing
        label = result['label']
        conf = result['confidence']
        score = result['sentiment_score']
        
        print(f"\nInput:  '{text}'")
        print(f"Result: {label} (Confidence: {conf:.1%})")
        print(f"Score:  {score:.4f}")

    print("\n" + "="*40)
    print("✅ SUCCESS: The package is working correctly!")

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: Could not find the package.\nDetails: {e}")
    print("Tip: Make sure 'climate_sentiment_pkg' has an empty '__init__.py' file inside it.")

except Exception as e:
    print(f"\n❌ RUNTIME ERROR: {e}")