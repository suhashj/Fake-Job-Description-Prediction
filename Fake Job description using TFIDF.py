#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string

# Load the dataset
df = pd.read_csv("fake_job_postings.csv")

# Display basic information about the dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

# Fill missing values with an empty string
df.fillna(" ", inplace=True)

# Visualization of fraudulent job postings
plt.figure(figsize=(4, 3))
sns.countplot(x='fraudulent', data=df)
plt.show()
print(df.fraudulent.value_counts())

# Combine relevant text columns into a single 'text' column
df['text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']

# Drop unnecessary columns
df.drop(['title', 'location', 'salary_range', 'employment_type', 'job_id', 'department', 
         'company_profile', 'description', 'requirements', 'benefits', 'required_experience', 
         'telecommuting', 'has_company_logo', 'has_questions', 'required_education', 
         'industry', 'function'], inplace=True, axis=1)

# Preprocessing: Convert text to lowercase
df['text'] = df['text'].str.lower()

# Tokenization and lemmatization
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)

df['text'] = df['text'].apply(lemmatize_words)

# Remove stopwords
stop_words = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Remove short words (length < 2)
df['clean_text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

# Generate word clouds for fraudulent and real job postings
fake_jobs = df[df["fraudulent"] == 1]["clean_text"]
real_jobs = df[df["fraudulent"] == 0]["clean_text"]

# Word cloud for fake jobs
wordcloud = WordCloud(max_font_size=80, max_words=30, background_color="red").generate(str(fake_jobs))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Word cloud for real jobs
wordcloud = WordCloud(max_font_size=80, max_words=30, background_color="gray").generate(str(real_jobs))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(df['clean_text'], df['fraudulent'], 
                                                    stratify=df['fraudulent'], 
                                                    test_size=0.3, random_state=123)

# Vectorization using TfidfVectorizer
Tfidf_vect = TfidfVectorizer(stop_words=stop_words, max_df=0.8, ngram_range=(1, 2), min_df=3)
Tfidf_vect.fit(train_X)

train_X_Tfidf = Tfidf_vect.transform(train_X)
test_X_Tfidf = Tfidf_vect.transform(test_X)

# Using Support Vector Classifier
svc = SVC(C=10, gamma=0.1, kernel='rbf', class_weight='balanced')
svc.fit(train_X_Tfidf, train_y)

# Predictions
pred_train1 = svc.predict(train_X_Tfidf)
pred_test1 = svc.predict(test_X_Tfidf)

# Accuracy scores
print("SVM Accuracy Score on Train set: ", accuracy_score(train_y, pred_train1) * 100)
print("SVM Accuracy Score on Validation set: ", accuracy_score(test_y, pred_test1) * 100)

# Evaluation metrics
print(classification_report(test_y, pred_test1))

def eval_metrics(actual, prediction):
    print("Accuracy Score: {}".format(accuracy_score(actual, prediction)))
    print("Recall Score: {}".format(recall_score(actual, prediction)))
    print("F1 Score: {}".format(f1_score(actual, prediction)))

eval_metrics(test_y, pred_test1)

# Additional evaluation metrics
print('Accuracy:', accuracy_score(test_y, pred_test1))
print('Precision:', precision_score(test_y, pred_test1))
print('Recall:', recall_score(test_y, pred_test1))
print('F1-Score:', f1_score(test_y, pred_test1))
print('\nConfusion Matrix:\n', confusion_matrix(test_y, pred_test1))
print('\nClassification Report:\n', classification_report(test_y, pred_test1))

# Cross-Validation
from sklearn.model_selection import KFold, cross_val_score

# Define cross-validation method to use
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model using cross-validation
scores = cross_val_score(svc, train_X_Tfidf, train_y, cv=cv, scoring='f1_macro')

# Print the average cross-validation score
print("Average F1 score:", np.mean(scores))



