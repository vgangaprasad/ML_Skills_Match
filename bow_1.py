"""

*************************************************************************************
Code to find the job posting matching the skill set

** Final step in getting the right posting is in progress but most of the code is complete

https://www.oreilly.com/content/how-do-i-compare-document-similarity-using-python/ 

Check the link above for more information

*************************************************************************************



"""
import json as j
import csv as csv
import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.stem import SnowballStemmer
import re
import gensim
from gensim import similarities

print(dir(gensim))

csv_data = None
rows = []
i =0 
with open('csv_ML.csv') as data_file:
    csv_data = csv.reader(data_file, delimiter=',', quotechar='|')
    data_list = [row[9] for row in csv_data]
#print ("Data List:")
#print(data_list)
print("Number of documents:",len(data_list))
print(data_list)

from nltk.tokenize import word_tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in data_list]
#print("Gen DOCS = ")
#print(gen_docs)


"""         i = i+1
        print(i)
        print(row[9])
        print(len(row[9]))
        rows.append(row) """


dictionary = gensim.corpora.Dictionary(gen_docs)
#print(dictionary[5])
#print(dictionary.token2id['road'])
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)

sims = gensim.similarities.Similarity('c:\\users\\aarth\desktop\sim_test',tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)
print(type(sims))

query_doc = [w.lower() for w in word_tokenize("Statistics Machine Learning Python Visualization")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

sims = sorted(enumerate(query_doc_tf_idf)) 
print(sims)
for i, s in sims:
    print(s, i, data_list[i])
    
