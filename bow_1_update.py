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

sims = gensim.similarities.Similarity('users\\denisc\desktop\sim_test',tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)
print(type(sims))

skill_one = input('What is your first skill? ')
skill_two = input('What is your second skill? ')
skill_three = input('What is your third skill? ')
skill_four = input('What is your fourth skill? ')

query_doc = [w.lower() for w in word_tokenize(f"{skill_one} {skill_two} {skill_three} {skill_four}")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

# sims = sorted(enumerate(query_doc_tf_idf)) 
# print(sims)
# for i, s in sims:
#     print(s, i, data_list[i])

print(sims)
max_match = (sims[query_doc_tf_idf].max())
print(max_match)

# list_match = sims[query_doc_tf_idf]
# print(list_match)

arr = np.array(sims[query_doc_tf_idf])
print(arr)

import heapq
input_list = arr
number_of_elements = 5
max_five = heapq.nlargest(number_of_elements, input_list)
print(max_five)

max_one = max_five[0]
max_two = max_five[1]
max_three = max_five[2]
max_four = max_five[3]
max_five = max_five[4]

for index, item in enumerate(sims[query_doc_tf_idf]):
    if max_one == item:
        index_one = index
        print(item, index_one)
    elif max_two == item:
        index_two = index
        print(item, index_two)
    elif max_three == item:
        index_three = index
        print(item, index_three)
    elif max_four == item:
        index_four = index
        print(item, index_four)
    elif max_five == item:
        index_five = index
        print(item, index_five)

df = pd.read_csv("csv_ML.csv")

print(df.iloc[index_one]['title'])
print(df.iloc[index_two]['title'])
print(df.iloc[index_three]['title'])
print(df.iloc[index_four]['title'])
print(df.iloc[index_five]['title'])

