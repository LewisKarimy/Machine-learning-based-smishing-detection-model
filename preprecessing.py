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
import traceback
# Step 1: Load and prepare the dataset
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
# Apply the remove_stopwords function to the 'TEXT' column
df_no_duplicates = df_no_duplicates['TEXT'].apply(remove_stopwords)
#print(df_no_duplicates)
# Encode labels into numerical values
df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x == 'ham' else 1)
#print(df['LABEL'] )
