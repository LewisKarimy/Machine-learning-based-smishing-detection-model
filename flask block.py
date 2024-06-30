# Assuming you have a trained model (clf) and a preprocessor (preprocessor)
# Make sure to replace 'your_model.pkl' and 'your_preprocessor.pkl' with the actual filenames

import pickle
import pandas as pd  # Assuming you are using pandas for data manipulation

# Load the trained model and preprocessor
with open('python_app.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('preprecessing.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Function to classify a message as smishing (0) or ham (1)
def classify_message(message):
    # Preprocess the message using the preprocessor
    preprocessed_message = preprocessor.transform([message])

    # Make the prediction using the trained model
    prediction = clf.predict(preprocessed_message)

    return prediction[0]

message_to_classify = input('message: ')   
result = classify_message(message_to_classify)
def classify_message(message)
        if result == 1:
           print("Smishing")
        else:
            print("ham")
    
    
    
