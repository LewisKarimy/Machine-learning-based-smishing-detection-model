import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import re
#nltk.download('punkt')
import spacy 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load and prepare the dataset
df = pd.read_csv("E:\e - book\Tech\pr\smdataset\Dataset_5971.csv") 
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
# Apply the remove_stopwords function to the 'TEXT' column
df_no_duplicates = df_no_duplicates['TEXT'].apply(remove_stopwords)
#print(df_no_duplicates)
# Encode labels into numerical values
df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x == 'ham' else 1)
#print(df['LABEL'] )

# Step 2: Feature extraction
X = df["TEXT"]
y = df["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5969)  # You can adjust the number of features
# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Assuming you have a 'label' column with values 'smishing' or 'ham' (0 for ham, 1 for smishing)
class_distribution = df['LABEL'].value_counts()
#print("Class Distribution:")
#print(class_distribution)

# Oversample the minority class (smishing)
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

# step 3: Model Selection
from sklearn.naive_bayes import MultinomialNB
# Create and train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier
from sklearn.metrics import accuracy_score, classification_report

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
naive_bayes.fit(X_train_tfidf, y_train)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(naive_bayes, param_grid, cv=5, scoring='accuracy')

# Perform hyperparameter optimization on the training data
grid_search.fit(X_train_tfidf, y_train)

# Get the best estimator from the hyperparameter optimization
best_naive_bayes = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_naive_bayes.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy}')

# Generate a classification report
#print(classification_report(y_test, y_pred))

# Display the confusion matrix - aids in analyzing model performance, identifying mis-classifications, and improving predictive accuracy.
conf_matrix = confusion_matrix(y_test, y_pred) #  a performance evaluation tool
#print('Confusion Matrix:')
#print(conf_matrix)

# Get the best hyperparameters from the optimization
best_params = grid_search.best_params_
#print('Best Hyperparameters:', best_params)

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def interface2():
    return render_template('interfacec2.html')





















