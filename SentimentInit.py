import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

log = pd.read_excel('Data_Preprocessing_Full.xlsx')

# dropping the rows in which one or more columns have null values
log.dropna(inplace=True)
log.shape

#%matplotlib inline
print('Percentage of responses')
print(round(log.polarity.value_counts(normalize=True)*100,2))
round(log.polarity.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.show()

independent_var = log.text
dependent_var = log.polarity

IV_train, IV_test, DV_train, DV_test = train_test_split(independent_var, dependent_var, test_size=0.1, random_state=225)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver = "lbfgs")

from sklearn.pipeline import Pipeline
model = Pipeline([('vectorizer',tvec),('classifier',clf2)])
model.fit(IV_train, DV_train)

from sklearn.metrics import confusion_matrix
predictions = model.predict(IV_test)
confusion_matrix(predictions, DV_test)

from sklearn.metrics import accuracy_score,precision_score, recall_score

print("Accuracy : ", accuracy_score(predictions,DV_test))
print("Precision : ", precision_score(predictions,DV_test, average = 'weighted'))
print("Recall : ", recall_score(predictions,DV_test, average = 'weighted'))

example = ["food is not that bad"]
result = model.predict(example)

print(result)