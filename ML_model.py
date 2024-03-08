#!/usr/bin/env python
# coding: utf-8

# In[105]:


# importing pandas as pd
import pandas as pd

# Making data frame from the csv file
df = pd.read_csv("collected_news.csv")

df


# In[106]:


#df_train_con['type'] = df_train_con['type'].map({'RK1':0, 'BHK1':1, 'BHK2':2, 'BHK3':3, 'BHK4':4, 'BHK4PLUS':5})
df['Genre'] = df['Genre'].map({"Entertainment" :0,"City": 1, "National" : 2 , "Sci-tech": 3, "Opinion" : 4, "Sports" : 5, "Business" : 6 })
df


# In[107]:


specific_rows = df.iloc[[113, 114, 115]]
print(specific_rows)


# In[108]:


#df_train_con['type'] = df_train_con['type'].map({'RK1':0, 'BHK1':1, 'BHK2':2, 'BHK3':3, 'BHK4':4, 'BHK4PLUS':5})
'''df['Segment'] = df['Segment'].map({
     "Art":0 , "Bangalore" : 1 ,"Chennai" : 2, "Coimbatore" : 3, "Delhi" : 4 ,"Hyderabad": 5,"Kochi" :6,"Kolkata" : 7,"Kozhikode" :8,
     "Madurai" : 9,"Mangalore" :10,"Mumbai": 11 ,"Puducherry" :12,"Thiruvananthapuram": 13,"Tiruchirapalli" :14 ,"Vijayawada":15 ,
     "Visakhapatnam": 16,"Andra Pradesh" :17,"Karnataka": 18,"Kerala" :19,"Tamil Nadu" :20,"Telangana" :21,"Other States":22,
     "Science":23,"Technology":24,"Health":25,"Agriculture" :26,"Energy-and-environment" :27,"Gadgets":28,"Internet":29,
    "Editorial" :30, "Columns" : 31,"Letters" :32,"Interview" :33,"Lead" :34, "Op-ed" :35,"cricket" : 36,"Football" : 37,"Hockey" : 38,"Tennis": 39,
    "Athletics":40,"Motorsport":41,"Races" :42,"Other-sports": 43,"Agri-business" :44,"Industry" :45, "Economy" : 46, "Markets":47, "Budget":48})
df
'''


# In[109]:


df.isnull().sum()


# In[110]:


#filtered_df = df[df['Segment'].isnull()]
#genre_and_title_with_null_segment = filtered_df[['Genre', 'Title']]
#print(genre_and_title_with_null_segment)
df


# In[ ]:





# In[111]:


"""import pandas as pd

# Assuming df["title"] contains a list of titles
data = {"title": df["Title"]}
df = pd.DataFrame(data)

# Define a function to remove special characters
def remove_special_characters(text):
    #return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'[^a-zA-Z0-9\s#,\|]', '', text)

# Apply the function to the "title" column
df['cleaned_title'] = df['title'].apply(remove_special_characters)

# Print the DataFrame with the cleaned title
print(df[['title', 'cleaned_title']])
"""


# In[112]:


df_dum = pd.get_dummies(df, columns=['Segment','Title'])
df_dum


# In[113]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming df is your DataFrame
corpus = df['Title'].tolist()

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Convert the TF-IDF matrix to a Pandas DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# One-hot encode categorical features
#categorical_columns = ['Genre', 'Segment']
#df_categorical = pd.get_dummies(df[categorical_columns])

# Concatenate the TF-IDF matrix and one-hot encoded categorical features
numerical_representation = pd.concat([tfidf_df, df_dum], axis=1)


# In[114]:


from sklearn.cluster import KMeans

# Assuming numerical_representation is your combined numerical representation
num_clusters = 6  # Choose the number of clusters based on the topics you want to identify

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(numerical_representation)  # Use .values for NumPy array or pass DataFrame directly


# In[117]:


# Naive Bayes

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Assuming df is your DataFrame containing the labeled clusters and topic labels
X_train, X_test, y_train, y_test = train_test_split(df['Title'], df['Genre'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_result)


# In[118]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# Assuming X_train_tfidf and y_train are the training data and labels
# and X_test_tfidf and y_test are the testing data and labels

# Train a Naive Bayes classifier (you can replace this with your actual model)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")


# In[119]:


# HyperTuning 
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}

# Create a Naive Bayes classifier
nb_classifier = MultinomialNB()

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best Parameters:", best_params)

# Use the best model for prediction
best_nb_classifier = grid_search.best_estimator_
y_pred_best = best_nb_classifier.predict(X_test_tfidf)


# In[121]:


# Performance Evaluvation 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Assuming y_test and y_pred are your true labels and predicted labels, respectively

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[127]:


#CROSS_VALIDATION FOR THE NAIVES BEYER MODEL

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform the entire dataset
X_tfidf = tfidf_vectorizer.fit_transform(df['Title'])

# Create the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Combine TF-IDF vectorizer and Naive Bayes classifier into a pipeline
model = make_pipeline(tfidf_vectorizer, nb_classifier)

# Define the cross-validation strategy (StratifiedKFold for classification tasks)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, df['Title'], df['Genre'], cv=cv, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# In[128]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Train the model on the entire dataset
model.fit(df['Title'], df['Genre'])

# Make predictions on the entire dataset
y_pred_all = model.predict(df['Title'])

# Create the confusion matrix
conf_matrix = confusion_matrix(df['Genre'], y_pred_all)

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df['Genre'].unique(), yticklabels=df['Genre'].unique())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




