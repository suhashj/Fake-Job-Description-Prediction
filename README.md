## Fake-Job_Description_prediction

# Project Overview
This project aims to detect fake job descriptions using machine learning techniques. We utilize Bi-LSTM (Bidirectional Long Short-Term Memory), Bag of Words, and TF-IDF (Term Frequency-Inverse Document Frequency) for text processing and classification. The goal is to build a predictive model that can distinguish between real and fake job postings based on textual data.

Table of Contents
Project Overview
Installation
Dataset
Preprocessing
Model
Training
Evaluation
Results


Dataset Preprocessing
Text Cleaning: We clean the text by removing unnecessary characters, stopwords, punctuation, and perform tokenization.

Feature Extraction:
Bag of Words (BoW): Converts the job descriptions into a matrix of token counts.

TF-IDF: Calculates the term frequency and inverse document frequency for the text data, providing a weighted representation of the words.

Preprocessing
Tokenization: Splitting the job description text into individual words.
Stopwords Removal: Removing common words that do not add much value to the classification process.
Padding: Adding padding to ensure that all input sequences are of the same length.
Encoding: Converting text to numeric form using the BoW or TF-IDF vectorization.
