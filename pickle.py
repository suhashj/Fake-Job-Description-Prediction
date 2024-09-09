import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('fake-final-classification.h5')

# Load the tokenizer
max_features = 10000
sent_length = 1370
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)

# Define the prediction function
def predict_fraudulent_job_posting(text):
    # Preprocess the input text
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)

    text = text.lower()
    text = text.strip()
    text = nltk.word_tokenize(text)
    text = [word for word in text if not word in set(nltk.corpus.stopwords.words("english"))]
    lemma = nltk.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    text = text.replace('  ',' ')
    
    # Tokenize and pad the input text
    tokenizer.fit_on_texts([text])
    encoded_docs = tokenizer.texts_to_sequences([text])
    padded_docs = pad_sequences(encoded_docs, padding='post', maxlen=sent_length)
    
    # Make a prediction using the model
    prediction = model.predict(padded_docs)
    
    # Return the prediction
    return prediction

# Define the Streamlit app
def main():
    st.title('Fraudulent Job Posting Detection')
    st.write('Enter a job posting description to check if it is fraudulent or not:')
    
    # Get user input
    input_text = st.text_area('Job Posting Description')
    
    # Make a prediction and display the result
    if st.button('Check'):
        if len(input_text) > 0:
            prediction = predict_fraudulent_job_posting(input_text)
            st.write(prediction)
            if prediction > 0.5:
                st.write("This job posting is **fraudulent**.")
            else:
                st.write("This job posting is **NOT Fraudulent**")

        else:
            st.write('Please enter a job posting description.')

# Run the app
if __name__ == '__main__':
    main()
