# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

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

# Vectorization using CountVectorizer
count_vect = CountVectorizer(stop_words=stop_words, max_df=0.8)
count_vect.fit(train_X)

train_X_count = count_vect.transform(train_X)
test_X_count = count_vect.transform(test_X)

# Using Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
rf.fit(train_X_count, train_y)

# Predictions
pred_train = rf.predict(train_X_count)
pred_test = rf.predict(test_X_count)

# Accuracy scores
print("Random Forest Accuracy Score on Train set: ", accuracy_score(train_y, pred_train) * 100)
print("Random Forest Accuracy Score on Validation set: ", accuracy_score(test_y, pred_test) * 100)
print(classification_report(test_y, pred_test))

# Hyperparameter tuning using GridSearchCV
rf = RandomForestClassifier(random_state=123)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(train_X_count, train_y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluation metrics
def eval_metrics(actual, prediction):
    print("Accuracy Score: {}".format(accuracy_score(actual, prediction)))
    print("Recall Score: {}".format(recall_score(actual, prediction)))
    print("F1 Score: {}".format(f1_score(actual, prediction)))

eval_metrics(test_y, pred_test)

# Additional evaluation metrics
print('Accuracy:', accuracy_score(test_y, pred_test))
print('Precision:', precision_score(test_y, pred_test))
print('Recall:', recall_score(test_y, pred_test))
print('F1-Score:', f1_score(test_y, pred_test))
print('\nConfusion Matrix:\n', confusion_matrix(test_y, pred_test))
print('\nClassification Report:\n', classification_report(test_y, pred_test))


