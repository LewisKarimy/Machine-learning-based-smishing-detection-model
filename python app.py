import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import re
#nltk.download('punkt')
import spacy 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 
import pickle
# Step 1: Preprocessing; Load and prepare the dataset
df = pd.read_csv("C:/Users/Karimi Lewis/my flask app/Dataset_5971.csv") 
#print(df.head(10))

# Function to clean text
def clean_TEXT(TEXT): 
    if TEXT is not None:  
        TEXT = re.sub(r'[^a-zA-Z\s]', '', TEXT)
        TEXT = TEXT.lower()
    return TEXT
df['TEXT'] = df['TEXT'].apply(clean_TEXT)  # Apply the clean_TEXT function to the 'TEXT' column
#print(df['TEXT'].head(20))

# removing dupicate TEXT
df = pd.DataFrame(df)
#print(df)
df_no_duplicates = df.drop_duplicates() # Check for and remove duplicate TEXT
#print(df_no_duplicates)

# Define a function to remove stopwords from a text
def remove_stopwords(TEXT):
    words = TEXT.split() # Tokenize the text into words
    words = [word for word in words if word.lower() not in stopwords.words('english')] # Remove stopwords
    # Join the remaining words back into a sentence
    return ' '.join(words)

df_no_duplicates = df_no_duplicates['TEXT'].apply(remove_stopwords)
#print(df_no_duplicates)

# Encode LABELS into numerical values
df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x == 'ham' else 1)
#print(df['LABEL'] )

# Step 2: Feature extraction
X = df["TEXT"]
y = df["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use CountVectorizer to convert text data into a feature matrix
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Assuming you have a 'label' column with values 'smishing' or 'ham' (0 for ham, 1 for smishing)
class_distribution = df['LABEL'].value_counts()
#print("Class Distribution:")
#print(class_distribution)

# Oversample the minority class (smishing)
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# step 3: Model Selection
from sklearn.naive_bayes import MultinomialNB
# Create and train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#print(f"Accuracy: {accuracy}")
#print(report)

#step 4 : Model training and evaluation
# Define a parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.1, 0.01, 0.001, 0.0001]  # Example values for hyperparameter tuning
}

# Initialize the Naive Bayes classifier
naive_bayes = MultinomialNB()
# Train the model on the training data
naive_bayes.fit(X_train, y_train)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(naive_bayes, param_grid, cv=5, scoring='accuracy')

# Perform hyperparameter optimization on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator from the hyperparameter optimization
best_naive_bayes = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_naive_bayes.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Generate a classification report
#print(classification_report(y_test, y_pred))

# Display the confusion matrix - aids in analyzing model performance, identifying mis-classifications, and improving predictive accuracy.
conf_matrix = confusion_matrix(y_test, y_pred) #  a performance evaluation tool
#print('Confusion Matrix:')
#print(conf_matrix)

# Get the best hyperparameters from the optimization
best_params = grid_search.best_params_
#print('Best Hyperparameters:', best_params)

# classifiaction and saving a module
# Save the trained model and vectorizer for future use
with open('python_app.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open('preprecessing.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    

# Now, when you want to classify a new message:
def classify_message(message):
    # Load the trained model and vectorizer
    with open('python_app.pkl', 'rb') as model_file:
        loaded_classifier = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    # Preprocess the new message using the same vectorizer
    message_vectorized = loaded_vectorizer.transform([message])

    # Use the loaded model to predict
    prediction = loaded_classifier.predict(message_vectorized)

    return prediction[0]  # Assuming prediction is an array with one element

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

with open('python_app.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('preprecessing.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)


def classify_message(message):
    preprocessed_message = preprocessor.transform([message])
    prediction = clf.predict(preprocessed_message)
    return "Smishing" if prediction[0] == 1 else "legitimate"

@app.route('/')
def frontend():
    return render_template('frontend.html')

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        message = data['message']

        result = classify_message(message)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)





















