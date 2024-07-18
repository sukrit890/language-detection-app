import sys
import joblib

def detect_language(text):
    # Load the trained model
    model = joblib.load('./python-scripts/language_detection_model.pkl')

    # Perform language detection
    language = model.predict([text])[0]

    return language

if __name__ == "__main__":
    # Check if text argument is provided
    if len(sys.argv) < 2:
        print("Please provide a text input.")
        sys.exit(1)
    
    # Read text input from Node.js command line argument
    text = sys.argv[1]

    # Perform language detection
    detected_language = detect_language(text)

    # Output the detected language
    print(detected_language)
