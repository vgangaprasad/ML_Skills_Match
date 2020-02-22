import json as j
import csv as csv
import sys
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from wordcloud import WordCloud


csv_data = None
rows = []
with open('csv_ML.csv') as data_file:
    csv_data = csv.reader(data_file, delimiter=',', quotechar='|')
    for row in csv_data:
        rows.append(row)

data = pd.read_csv("csv_ML.csv")
data.head()
print(data['description'].head())


""" data = pd.DataFrame(json_data) """

stemmer = SnowballStemmer('english')
words = stopwords.words("english")

data['cleaned'] = data['description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

X_train, X_test, y_train, y_test = train_test_split(data['cleaned'],data.title, test_size=0.2)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = ['1', '2', '3', '4', '5']
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top100 = np.argsort(clf.coef_[i])[-100:]
    print("%s: %s" % (label, " ".join(feature_names[top100])))

print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['Keras Deep Learning. Logistic food!']))
