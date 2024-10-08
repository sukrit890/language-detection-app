import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from wordcloud import WordCloud

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    st.warning("Plotly is not installed. Install it to use interactive visualizations.")

try:
    import seaborn as sns
except ImportError:
    st.warning("Seaborn is not installed. Some visualizations may not work properly.")

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

# Function to plot pie chart
def plot_pie_chart(language_counts):
    if not language_counts.empty:
        fig = px.pie(language_counts, names=language_counts.index, values=language_counts.values, 
                     title='Language Distribution in Batch Results',
                     color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig)
    else:
        st.warning("No data available for pie chart visualization.")

# Function to plot interactive bar chart
def plot_interactive_bar_chart(language_counts):
    if not language_counts.empty:
        fig = go.Figure([go.Bar(x=language_counts.index, y=language_counts.values, 
                               marker_color='lightsalmon')])
        fig.update_layout(title='Language Distribution in Batch Results',
                          xaxis_title='Languages',
                          yaxis_title='Counts')
        st.plotly_chart(fig)
    else:
        st.warning("No data available for interactive bar chart.")

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
                st.success(f'Detected Language: {detected_language}')
                st.write(f'Confidence: {confidence:.2f}')
            else:
                st.warning(f'Confidence below threshold. Detected Language: {detected_language}')
                st.write(f'Confidence: {confidence:.2f}')
        else:
            st.warning("Please enter some text.")

    # Text input box for batch detection
    st.header('Batch Text Detection')
    batch_texts = st.text_area('Enter multiple texts for batch detection (one per line):', '').split('\n')

    # Language selection dropdown for batch detection
    selected_language_batch = st.selectbox('Select language to detect for batch:', languages)

    # Detect languages for batch texts on button click
    if st.button('Detect Languages (Batch Texts)'):
        if batch_texts:
            results = detect_languages_batch(batch_texts, selected_language_batch)
            if results:
                st.write(f'Batch Detection Results:')
                for text, lang, conf in results:
                    st.write(f'Text: {text}')
                    st.write(f'Detected Language: {lang}')
                    st.write(f'Confidence: {conf:.2f}')
                    st.write('---')

                # Create word cloud
                st.subheader('Word Cloud')
                create_word_cloud(results)

                # Plot pie chart
                st.subheader('Pie Chart')
                language_counts = pd.Series([lang for _, lang, _ in results]).value_counts()
                plot_pie_chart(language_counts)

                # Plot interactive bar chart
                st.subheader('Interactive Bar Chart')
                plot_interactive_bar_chart(language_counts)

            else:
                st.warning("No results available for batch detection.")
        else:
            st.warning("Please enter texts for batch detection.")

if __name__ == "__main__":
    main()
