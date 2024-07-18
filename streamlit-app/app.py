import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from wordcloud import WordCloud

# Function to detect language given text and selected language
def detect_language(text, selected_language):
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../python-scripts/language_detection_model.pkl')
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None, None

    if text:
        try:
            if selected_language == 'Auto Detect':
                confidence_scores = model.predict_proba([text])[0]
                detected_language = model.classes_[np.argmax(confidence_scores)]
                confidence = max(confidence_scores)
            else:
                detected_language = selected_language
                confidence_scores = model.predict_proba([text])[0]
                language_index = list(model.classes_).index(selected_language)
                confidence = confidence_scores[language_index]

            return detected_language, confidence

        except Exception as e:
            st.error(f"Error detecting language: {str(e)}")
            return None, None
    else:
        st.warning("Please enter some text for language detection.")
        return None, None

# Function to detect languages for multiple texts
def detect_languages_batch(texts, selected_language):
    results = []
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../python-scripts/language_detection_model.pkl')
        model = joblib.load(model_path)

        for text in texts:
            try:
                if selected_language == 'Auto Detect':
                    detected_language = model.predict([text])[0]
                    confidence_scores = model.predict_proba([text])[0]
                    confidence = max(confidence_scores)
                else:
                    detected_language = selected_language
                    confidence_scores = model.predict_proba([text])[0]
                    language_index = list(model.classes_).index(selected_language)
                    confidence = confidence_scores[language_index]

                results.append((text, detected_language, confidence))

            except Exception as e:
                st.warning(f"Error processing text: {str(e)}")
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

    return results

# Function to create a word cloud
def create_word_cloud(results):
    text_data = ' '.join([text for text, _, _ in results])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Main function to create the Streamlit app
def main():
    # Custom HTML for background styling
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #1a1a1a, #2e3a4f);
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            color: #ffffff;
            text-align: center;
            padding: 30px 0;
            font-size: 2.5em;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #007bff;
            color: #ffffff;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #444444;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextArea>div>textarea {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #444444;
            border-radius: 8px;
            padding: 10px;
        }
        .stMarkdown {
            color: #e1e1e1;
        }
        .stSelectbox>div>div>select {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #444444;
            border-radius: 8px;
            padding: 10px;
        }
        .stSlider>div>div>input {
            accent-color: #007bff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Language Detection App')

    # Text input box for single text detection
    st.header('Single Text Detection')
    text_input = st.text_area('Enter text to detect language:', '')

    # Language selection dropdown for single text detection
    languages = ['Auto Detect', 'English', 'French', 'German', 'Spanish', 'Italian', 'Chinese', 'Japanese']
    selected_language_single = st.selectbox('Select language to detect:', languages)

    # Confidence threshold slider
    confidence_threshold = st.slider('Minimum Confidence Level', min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # Detect language for single text on button click
    if st.button('Detect Language (Single Text)'):
        if text_input:
            detected_language, confidence = detect_language(text_input, selected_language_single)
            if confidence is None:
                st.success(f'Detected Language: {detected_language}')
            elif confidence >= confidence_threshold:
                st.success(f'Detected Language: {detected_language} (Confidence: {confidence:.2f})')
            else:
                st.warning(f'Confidence below threshold ({confidence_threshold:.2f}): No language detected.')

    # Text input box for batch processing
    st.header('Batch Processing')
    batch_text_input = st.text_area('Enter multiple texts (one per line) for batch processing:', '')

    # Language selection dropdown for batch processing
    selected_language_batch = st.selectbox('Select language to detect for batch processing:', languages)

    # Detect languages for batch processing on button click
    if st.button('Detect Languages (Batch Processing)'):
        if batch_text_input:
            texts = batch_text_input.split('\n')
            results = detect_languages_batch(texts, selected_language_batch)

            # Display batch processing results
            st.subheader('Batch Processing Results:')
            if results:
                for text, detected_language, confidence in results:
                    if confidence is None:
                        st.write(f'Text: "{text}" - Detected Language: {detected_language}')
                    elif confidence >= confidence_threshold:
                        st.write(f'Text: "{text}" - Detected Language: {detected_language} (Confidence: {confidence:.2f})')
                    else:
                        st.write(f'Text: "{text}" - Confidence below threshold ({confidence_threshold:.2f}): No language detected.')

                # Interactive visualization: Language distribution
                st.header('Interactive Visualization')
                df_results = pd.DataFrame(results, columns=['Text', 'Language', 'Confidence'])
                language_counts = df_results['Language'].value_counts()

                plt.figure(figsize=(10, 6))
                plt.bar(language_counts.index, language_counts.values, color='#007bff')
                plt.xlabel('Languages')
                plt.ylabel('Counts')
                plt.title('Language Distribution in Batch Results')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(plt)

                # Word cloud visualization
                st.header('Word Cloud')
                create_word_cloud(results)

if __name__ == '__main__':
    main()
