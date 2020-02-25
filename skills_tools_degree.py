#load the data
import csv as csv
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import pos_tag
from keywords import tool_keywords1
from keywords import tool_keywords2
from keywords import skill_keywords1
from keywords import skill_keywords2
from keywords import degree_dict
from keywords import degree_dict2
from keywords import degree_keywords2

df = pd.read_csv("csv_ML.csv")
ps = PorterStemmer()


# process the job description.
def prepare_job_desc(desc):
    # tokenize description.
    tokens = word_tokenize(desc)
        
    # Parts of speech (POS) tag tokens.
    token_tag = pos_tag(tokens)
    
    # Only include some of the POS tags.
    include_tags = ['VBN', 'VBD', 'JJ', 'JJS', 'JJR', 'CD', 'NN', 'NNS', 'NNP', 'NNPS']
    filtered_tokens = [tok.lower() for tok, tag in token_tag if tag in include_tags]
    
    # stem words.
    stemmed_tokens = [ps.stem(tok).lower() for tok in filtered_tokens]
    #return set(stemmed_tokens)
    return set(filtered_tokens)

df['job_description_word_set'] = df['description'].map(prepare_job_desc)

tool_keywords1_set = set([tok for tok in tool_keywords1]) # stem the keywords (since the job description is also stemmed.)
tool_keywords1_dict = {tok:tok for tok in tool_keywords1} # use this dictionary to revert the stemmed words back to the original.

skill_keywords1_set = set([tok for tok in skill_keywords1])
skill_keywords1_dict = {tok:tok for tok in skill_keywords1}

degree_keywords1_set = set([tok for tok in degree_dict.keys()])
degree_keywords1_dict = {tok:tok for tok in degree_dict.keys()}

tool_list = []
skill_list = []
degree_list = []

msk = df['title'] != '' # just in case you want to filter the data.
#print("msk ")
#print(msk)
num_postings = len(df[msk].index)

for i in range(num_postings):
    job_desc = df[msk].iloc[i]['description'].lower()
    job_desc_set = df[msk].iloc[i]['job_description_word_set']
    
    # check if the keywords are in the job description. Look for exact match by token.
    tool_words = tool_keywords1_set.intersection(job_desc_set)
    skill_words = skill_keywords1_set.intersection(job_desc_set)
    degree_words = degree_keywords1_set.intersection(job_desc_set)
    
    # check if longer keywords (more than one word) are in the job description. Match by substring.
    j = 0
    for tool_keyword2 in tool_keywords2:
        # tool keywords.
        if tool_keyword2 in job_desc:
            tool_list.append(tool_keyword2)
            j += 1
    
    k = 0
    for skill_keyword2 in skill_keywords2:
        # skill keywords.
        if skill_keyword2 in job_desc:
            skill_list.append(skill_keyword2)
            k += 1
    
    # search for the minimum education.
    min_education_level = 999
    for degree_word in degree_words:
        level = degree_dict[degree_keywords1_dict[degree_word]]
        min_education_level = min(min_education_level, level)
    
    for degree_keyword2 in degree_keywords2:
        # longer keywords. Match by substring.
        if degree_keyword2 in job_desc:
            level = degree_dict2[degree_keyword2]
            min_education_level = min(min_education_level, level)
    
    # label the job descriptions without any tool keywords.
    if len(tool_words) == 0 and j == 0:
        tool_list.append('nothing specified')
    
    # label the job descriptions without any skill keywords.
    if len(skill_words) == 0 and k == 0:
        skill_list.append('nothing specified')
    
    # If none of the keywords were found, but the word degree is present, then assume it's a bachelors level.
    if min_education_level > 500:
        if 'degree' in job_desc:
            min_education_level = 1
    
    tool_list += list(tool_words)
    skill_list += list(skill_words)
    degree_list.append(min_education_level)

# create the list of tools.
df_tool = pd.DataFrame(data={'cnt': tool_list})
df_tool = df_tool.replace(tool_keywords1_dict)

# group some of the categories together.
msk = df_tool['cnt'] == 'h20'
df_tool.loc[msk, 'cnt'] = 'h2o'

msk = df_tool['cnt'] == 'aws'
df_tool.loc[msk, 'cnt'] = 'amazon web services'

msk = df_tool['cnt'] == 'gcp'
df_tool.loc[msk, 'cnt'] = 'google cloud'

msk = df_tool['cnt'] == 'github'
df_tool.loc[msk, 'cnt'] = 'git'

msk = df_tool['cnt'] == 'postgressql'
df_tool.loc[msk, 'cnt'] = 'postgres'

msk = df_tool['cnt'] == 'tensor'
df_tool.loc[msk, 'cnt'] = 'tensorflow'

df_tool_top25 = df_tool['cnt'].value_counts().reset_index().rename(columns={'index': 'tool'}).iloc[:25]

df_tool_top50 = df_tool['cnt'].value_counts().reset_index().rename(columns={'index': 'tool'}).iloc[:50]

# create the list of skills/knowledge.
df_skills = pd.DataFrame(data={'cnt': skill_list})
df_skills = df_skills.replace(skill_keywords1_dict)

# group some of the categories together.
msk = df_skills['cnt'] == 'nlp'
df_skills.loc[msk, 'cnt'] = 'natural language processing'

msk = df_skills['cnt'] == 'convolutional'
df_skills.loc[msk, 'cnt'] = 'convolutional neural network'

msk = df_skills['cnt'] == 'cnn'
df_skills.loc[msk, 'cnt'] = 'convolutional neural network'

msk = df_skills['cnt'] == 'recurrent'
df_skills.loc[msk, 'cnt'] = 'recurrent neural network'

msk = df_skills['cnt'] == 'rnn'
df_skills.loc[msk, 'cnt'] = 'recurrent neural network'

msk = df_skills['cnt'] == 'knn'
df_skills.loc[msk, 'cnt'] = 'nearest neighbors'

msk = df_skills['cnt'] == 'svm'
df_skills.loc[msk, 'cnt'] = 'support vector machine'

msk = df_skills['cnt'] == 'machine vision'
df_skills.loc[msk, 'cnt'] = 'computer vision'

msk = df_skills['cnt'] == 'ab testing'
df_skills.loc[msk, 'cnt'] = 'a/b testing'

df_skills_top50 = df_skills['cnt'].value_counts().reset_index().rename(columns={'index': 'skill'}).iloc[:50]

df_skills_top25 = df_skills['cnt'].value_counts().reset_index().rename(columns={'index': 'skill'}).iloc[:25]

df_new_temp = pd.DataFrame(data={'keyword' : tool_list})
df_new = pd.DataFrame(data={'keyword' :df_new_temp['keyword'].unique()})
df_new.append(df_skills,'sort=True')

print(df_new)

# create the list of degree.
df_degrees = pd.DataFrame(data={'cnt': degree_list})
df_degrees['degree_type'] = ''


msk = df_degrees['cnt'] == 1
df_degrees.loc[msk, 'degree_type'] = 'bachelors'

msk = df_degrees['cnt'] == 2
df_degrees.loc[msk, 'degree_type'] = 'masters'

msk = df_degrees['cnt'] == 3
df_degrees.loc[msk, 'degree_type'] = 'phd'

msk = df_degrees['cnt'] == 4
df_degrees.loc[msk, 'degree_type'] = 'postdoc'

msk = df_degrees['cnt'] == 2.5
df_degrees.loc[msk, 'degree_type'] = 'mba'

msk = df_degrees['cnt'] > 500
df_degrees.loc[msk, 'degree_type'] = 'not specified'


df_degree_cnt = df_degrees['degree_type'].value_counts().reset_index().rename(columns={'index': 'degree'}).iloc[:50]
df_degree_cnt
