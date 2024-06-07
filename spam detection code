import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load stopwords
nltk.download('stopwords')

def include_csv(filename):
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''

    # 1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2
    clean_words = [word for word in nopunc.split() if word.lower() not in stop_words]

    # 3
    return clean_words

# Load stopwords outside the function
stop_words = set(stopwords.words('english'))

# Read CSV file
messages = pd.read_csv("C:\\Users\\bilaa\\OneDrive\\Documents\\spam.csv", encoding='latin-1')
messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
messages = messages.rename(columns={'v1': 'class','v2': 'text'})
messages.groupby('class').describe()
messages['length'] = messages['text'].apply(len)
messages.hist(column='length',by='class',bins=50, figsize=(15,6))

msg_train, msg_test, class_train, class_test = train_test_split(messages['text'],messages['class'],test_size=0.2)
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=process_text)), # converts strings to integer counts
    ('tfidf',TfidfTransformer()), # converts integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB()) # train on TF-IDF vectors with Naive Bayes classifier
])
pipeline.fit(msg_train,class_train)
predictions = pipeline.predict(msg_test)
print(classification_report(class_test,predictions))
def classify_user_input():
    input_text = input("Enter a text to check if it's spam or ham: ")
    prediction = pipeline.predict([input_text])
    if prediction[0] == 'ham':
        print("The input text is not spam.")
    else:
        print("The input text is spam.")

if __name__ == "__main__":
    classify_user_input()
