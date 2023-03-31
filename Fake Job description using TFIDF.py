#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from nltk.corpus import stopwords


# In[2]:


df=pd.read_csv("fake_job_postings.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.head()


# In[9]:


df.fillna(" ",inplace=True)


# In[10]:


#visualization
plt.figure(figsize = (4,3))
sns.countplot(x='fraudulent',data=df)
plt.show()


# In[11]:


df.fraudulent.value_counts()


# In[12]:


df


# In[13]:


df.required_education.value_counts()


# In[14]:


df['text']= df['title']+" " + df['company_profile']+ " " +df['description']+ " " + df['requirements']+ " " + df['benefits']


# In[15]:


df


# In[16]:


df.drop(['title','location','salary_range','employment_type','job_id','department','company_profile','description','requirements','benefits','required_experience','telecommuting','has_company_logo','has_questions','required_education','industry','function'],inplace=True,axis=1)


# In[17]:


df.head()


# # PreProcessing

# In[18]:


punctuation=string.punctuation


# In[19]:


import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


# In[20]:


df.columns


# In[21]:


df['text'] = [sentence.lower() for sentence in df['text']]


# In[22]:


print(df['text'])


# In[23]:


# tokenize the text in the 'text_column' column
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 


tokenizer = RegexpTokenizer(r'\w+')
df["text"].apply(tokenizer.tokenize)


# In[24]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    words=text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    return ' '.join(words)
df['text'] = df['text'].apply(lemmatize_words)


# In[25]:


print(df['text'])


# In[26]:


stop_words = stopwords.words('english')

# Remove the stopwords from the 'text' column
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Print the DataFrame with stopwords removed
print(df)


# In[27]:


# remove short words (length < 2)
df['clean_text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))


# In[28]:


df.head()


# In[29]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[30]:


fake_jobs = df[df["fraudulent"] == 1]["clean_text"]
real_jobs = df[df["fraudulent"] == 0]["clean_text"]


# In[31]:


wordcloud = WordCloud(max_font_size=80,max_words=30, background_color="red").generate(str(fake_jobs))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[32]:


wordcloud = WordCloud(max_font_size=80,max_words=30, background_color="gray").generate(str(real_jobs))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[33]:


#define the variables


train_X, test_X, train_y, test_y = train_test_split(df['clean_text'], df['fraudulent'], 
                                                    stratify = df['fraudulent'],
                                                    test_size=0.3, random_state=123)


# In[34]:


print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)


# In[35]:


train_X.head()


# In[36]:


train_y.head()


# In[37]:


test_X.head()


# In[38]:


test_y.head()


# In[39]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer



Tfidf_vect = TfidfVectorizer(stop_words=stop_words, max_df=0.8, ngram_range=(1,2), min_df=3)

Tfidf_vect.fit(train_X)

train_X_Tfidf = Tfidf_vect.transform(train_X)
test_X_Tfidf = Tfidf_vect.transform(test_X)


# In[ ]:





# In[40]:


# # Coverting to dense matrix and putting in a dataframe to view the Tfidf matrix
# dense_mat = train_X_Tfidf.todense()
# tfidf_Mat = pd.DataFrame(dense_mat, columns=Tfidf_vect.get_feature_names_out())
# tfidf_Mat.head()


# In[41]:


# print([col for col in tfidf_Mat.columns])


# In[42]:


from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
import sklearn

svc = SVC(C=10, gamma=0.1, kernel='rbf', class_weight='balanced')

svc.fit(train_X_Tfidf,train_y)

# predict the labels on train dataset
pred_train1 = svc.predict(train_X_Tfidf)

# predict the labels on validation dataset
pred_test1 = svc.predict(test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score on Train set.     -> ", accuracy_score(train_y, pred_train1)*100)
print("SVM Accuracy Score on Validation set -> ", accuracy_score(test_y, pred_test1)*100)


# # Hyperparameter Tuning

# In[43]:


# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC

# # Create an SVM classifier object
# svc = SVC()

# # Define the hyperparameter grid to search
# param_grid = {
#     'C': [0.1, 1, 10, 100], 
#     'kernel': ['linear', 'rbf'], 
#     'gamma': ['scale', 'auto', 0.1, 1]
# }

# # Create a GridSearchCV object
# grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1_macro')

# # Fit the GridSearchCV object to the data
# grid_search.fit(train_X_Tfidf, train_y)

# # Print the best hyperparameters found
# print("Best hyperparameters:", grid_search.best_params_)

# # Print the best cross-validation score found
# print("Best cross-validation score:", grid_search.best_score_)


# In[44]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
print(classification_report(test_y, pred_test1))


# In[45]:


def eval_metrics(actual, prediction):
    print("Accuracy Score: {}".format(accuracy_score(actual, prediction)))
    print("Recall Score: {}".format(recall_score(actual, prediction)))
    print("f1 Score: {}".format(f1_score(actual, prediction)))


# In[46]:


eval_metrics(test_y, pred_test1)


# In[47]:


# Evaluation Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print('Accuracy:', accuracy_score(test_y, pred_test1))
print('Precision:', precision_score(test_y, pred_test1))
print('Recall:', recall_score(test_y, pred_test1))
print('F1-Score:', f1_score(test_y, pred_test1))
print('\nConfusion Matrix:\n', confusion_matrix(test_y, pred_test1))
print('\nClassification Report:\n', classification_report(test_y, pred_test1))



# # Cross-Validation

# In[48]:


from sklearn.model_selection import KFold, cross_val_score

# Define cross-validation method to use
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model using cross-validation
scores = cross_val_score(svc, train_X_Tfidf, train_y, cv=cv, scoring='f1_macro')

# Print the average cross-validation score
print("Average F1 score:", np.mean(scores))


# In[ ]:





# In[ ]:





# In[ ]:




