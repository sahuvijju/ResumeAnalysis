#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\HP\Downloads\Resume Screening1.csv")
print(df)


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop_duplicates().any()


# In[9]:


df.shape


# In[10]:


df["Category"].value_counts()


# In[11]:


df.corr()


# In[12]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
ax=sns.countplot(x="Category", data=df)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.show()


# In[13]:


plt.figure(figsize=(15,5))
sns.scatterplot(df["Category"])
plt.xticks(rotation=90)
plt.show()


# In[14]:


df['Category'].unique()


# In[15]:


counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()


# In[16]:


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# In[17]:


df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')


# In[20]:


tfidf.fit(df['Resume'])
requiredText = tfidf.transform(df['Resume'])


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)


# In[22]:


X_train


# In[23]:


X_test


# In[24]:


y_train


# In[25]:


y_test


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,classification_report


# In[27]:


clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)


# In[28]:


ypred = clf.predict(X_test)
ypred


# In[29]:


print(accuracy_score(y_test, ypred))


# In[30]:


print(classification_report(y_test, ypred))


# In[31]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(requiredText, df['Category'], test_size = 0.20, random_state = 0)


# In[32]:


X_train1


# In[33]:


X_test1


# In[34]:


y_train1


# In[35]:


y_test1


# In[36]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


# In[37]:


models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression' : LogisticRegression(),
    'Decision Tree' : DecisionTreeClassifier(),
    'NB' : GaussianNB(),
    'OKNN': OneVsRestClassifier(KNeighborsClassifier()),
    'KM': KMeans(),
     
}


# In[38]:


for model_name, model in models.items():
    model.fit(X_train1, y_train1)
    y_pred1 = model.predict(X_test1)
    accuracy = accuracy_score(y_test1, y_pred1)
    classification_rep = classification_report(y_test1, y_pred1)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_rep)
    print("=" * 50)

