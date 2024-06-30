import pickle  # For saving and loading the trained model
from sklearn.feature_extraction.text import CountVectorizer  # For text representation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Choose an appropriate classifier
import pandas as pd
# Assuming you have a dataset with features (X) and labels (y)
# X should be a list of text messages, and y should be the corresponding labels (1 for smishing, 0 for not smishing)

# Split the dataset into training and testing sets
df = pd.read_csv("C:/Users/Karimi Lewis/my flask app/Dataset_5971.csv") 
X = df["TEXT"]
y = df["LABEL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use CountVectorizer to convert text data into a feature matrix
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train your model (choose an appropriate classifier)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Save the trained model and vectorizer for future use
with open('smishing_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Now, when you want to classify a new message:
def classify_message(message):
    # Load the trained model and vectorizer
    with open('smishing_model.pkl', 'rb') as model_file:
        loaded_classifier = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    # Preprocess the new message using the same vectorizer
    message_vectorized = loaded_vectorizer.transform([message])

    # Use the loaded model to predict
    prediction = loaded_classifier.predict(message_vectorized)

    return prediction[0]  # Assuming prediction is an array with one element

# Example usage
new_message = "Congratulations! You've won a free gift. Click the link to claim."   
result = classify_message(new_message)

if result == 1:
    print("The message is classified as smishing.")
else:
    print("The message is not classified as smishing.")
    
 



