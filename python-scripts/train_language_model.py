import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

# Load dataset from CSV
df = pd.read_csv('./dataset.csv')  # Replace with actual path

# Assuming your CSV has 'text' and 'language' columns
X = df['Text'].values
y = df['language'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for feature extraction and model training
model_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Fit the pipeline
model_pipeline.fit(X_train, y_train)

# Predictions on test set
y_pred = model_pipeline.predict(X_test)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy}')

# Classification report on test set
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model_pipeline, './python-scripts/language_detection_model.pkl')
