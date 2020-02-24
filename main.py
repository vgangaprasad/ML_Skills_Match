#load the data
import csv as csv
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.stem import PorterStemmer

df = pd.read_csv("csv_ML.csv")

#list of keywords
# got these keywords by looking at some examples and using existing knowledge.
tool_keywords1 = ['python', 'pytorch', 'sql', 'mxnet', 'mlflow', 'einstein', 'theano', 'pyspark', 'solr', 'mahout', 
 'cassandra', 'aws', 'powerpoint', 'spark', 'pig', 'sas', 'java', 'nosql', 'docker', 'salesforce', 'scala', 'r',
 'c', 'c++', 'net', 'tableau', 'pandas', 'scikitlearn', 'sklearn', 'matlab', 'scala', 'keras', 'tensorflow', 'clojure',
 'caffe', 'scipy', 'numpy', 'matplotlib', 'vba', 'spss', 'linux', 'azure', 'cloud', 'gcp', 'mongodb', 'mysql', 'oracle', 
 'redshift', 'snowflake', 'kafka', 'javascript', 'qlik', 'jupyter', 'perl', 'bigquery', 'unix', 'react',
 'scikit', 'powerbi', 's3', 'ec2', 'lambda', 'ssrs', 'kubernetes', 'hana', 'spacy', 'tf', 'django', 'sagemaker',
 'seaborn', 'mllib', 'github', 'git', 'elasticsearch', 'splunk', 'airflow', 'looker', 'rapidminer', 'birt', 'pentaho', 
 'jquery', 'nodejs', 'd3', 'plotly', 'bokeh', 'xgboost', 'rstudio', 'shiny', 'dash', 'h20', 'h2o', 'hadoop', 'mapreduce', 
 'hive', 'cognos', 'angular', 'nltk', 'flask', 'node', 'firebase', 'bigtable', 'rust', 'php', 'cntk', 'lightgbm', 
 'kubeflow', 'rpython', 'unixlinux', 'postgressql', 'postgresql', 'postgres', 'hbase', 'dask', 'ruby', 'julia', 'tensor',
# added r packages doesn't seem to impact the result
 'dplyr','ggplot2','esquisse','bioconductor','shiny','lubridate','knitr','mlr','quanteda','dt','rcrawler','caret','rmarkdown',
 'leaflet','janitor','ggvis','plotly','rcharts','rbokeh','broom','stringr','magrittr','slidify','rvest',
 'rmysql','rsqlite','prophet','glmnet','text2vec','snowballc','quantmod','rstan','swirl','datasciencer']


# another set of keywords that are longer than one word.
tool_keywords2 = set(['amazon web services', 'google cloud', 'sql server'])

# hard skills/knowledge required.
skill_keywords1 = set(['statistics', 'cleansing', 'chatbot', 'cleaning', 'blockchain', 'causality', 'correlation', 'bandit', 'anomaly', 'kpi',
 'dashboard', 'geospatial', 'ocr', 'econometrics', 'pca', 'gis', 'svm', 'svd', 'tuning', 'hyperparameter', 'hypothesis',
 'salesforcecom', 'segmentation', 'biostatistics', 'unsupervised', 'supervised', 'exploratory',
 'recommender', 'recommendations', 'research', 'sequencing', 'probability', 'reinforcement', 'graph', 'bioinformatics',
 'chi', 'knn', 'outlier', 'etl', 'normalization', 'classification', 'optimizing', 'prediction', 'forecasting',
 'clustering', 'cluster', 'optimization', 'visualization', 'nlp', 'c#',
 'regression', 'logistic', 'nn', 'cnn', 'glm',
 'rnn', 'lstm', 'gbm', 'boosting', 'recurrent', 'convolutional', 'bayesian',
 'bayes'])


# another set of keywords that are longer than one word.
skill_keywords2 = set(['random forest', 'natural language processing', 'machine learning', 'decision tree', 'deep learning', 'experimental design',
 'time series', 'nearest neighbors', 'neural network', 'support vector machine', 'computer vision', 'machine vision', 'dimensionality reduction', 
 'text analytics', 'power bi', 'a/b testing', 'ab testing', 'chat bot', 'data mining'])

degree_dict = {'bs': 1, 'bachelor': 1, 'undergraduate': 1, 
               'master': 2, 'graduate': 2, 'mba': 2.5, 
               'phd': 3, 'ph.d': 3, 'ba': 1, 'ma': 2,
               'postdoctoral': 4, 'postdoc': 4, 'doctorate': 3}


degree_dict2 = {'advanced degree': 2, 'ms or': 2, 'ms degree': 2, '4 year degree': 1, 'bs/': 1, 'ba/': 1,
                '4-year degree': 1, 'b.s.': 1, 'm.s.': 2, 'm.s': 2, 'b.s': 1, 'phd/': 3, 'ph.d.': 3, 'ms/': 2,
                'm.s/': 2, 'm.s./': 2, 'msc/': 2, 'master/': 2, 'master\'s/': 2, 'bachelor\s/': 1}
degree_keywords2 = set(degree_dict2.keys())


#pos_tag(tool_keywords1)

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

# process the keywords
#tool_keywords1_set = set([ps.stem(tok) for tok in tool_keywords1]) # stem the keywords (since the job description is also stemmed.)
#tool_keywords1_dict = {ps.stem(tok):tok for tok in tool_keywords1} # use this dictionary to revert the stemmed words back to the original.
tool_keywords1_set = set([tok for tok in tool_keywords1]) # stem the keywords (since the job description is also stemmed.)
tool_keywords1_dict = {tok:tok for tok in tool_keywords1} # use this dictionary to revert the stemmed words back to the original.

#print("tool Keywords set:")
#print(tool_keywords1_set)
#print("Tool Keywords Dict:")
#print(tool_keywords1_dict)

#skill_keywords1_set = set([ps.stem(tok) for tok in skill_keywords1])
#skill_keywords1_dict = {ps.stem(tok):tok for tok in skill_keywords1}
skill_keywords1_set = set([tok for tok in skill_keywords1])
skill_keywords1_dict = {tok:tok for tok in skill_keywords1}

#degree_keywords1_set = set([ps.stem(tok) for tok in degree_dict.keys()])
#degree_keywords1_dict = {ps.stem(tok):tok for tok in degree_dict.keys()}
degree_keywords1_set = set([tok for tok in degree_dict.keys()])
degree_keywords1_dict = {tok:tok for tok in degree_dict.keys()}

tool_list = []
skill_list = []
degree_list = []

msk = df['title'] != '' # just in case you want to filter the data.
print("msk ")
print(msk)
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
#df_new = df_new_temp.unique()
df_new = pd.DataFrame(data={'keyword' :df_new_temp['keyword'].unique()})
#df_new['keyword'].value_counts()
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

#skills_list = "Statistics bayesian Machine Learning python Visualization"
skills_list = "EXPERIENCE Los Angeles, CA SOFTWARE ENGINEER, MACHINE LEARNING 08/2014 – present Work with Data Scientists and Product Managers to frame a problem, both mathematically and within the business context Deploy validated algorithms to our RTB system, and develop techniques for monitoring and visualizing performance of all deployed algorithms Knowledge developing and debugging in C/C++ and Java Knowledge of one or more open-source Machine Learning framework Develop prototypes and validate the results Expert knowledge developing and debugging in C/C++ and Java Contribute to the production solutions' development, testing and deployment Phoenix, AZ MACHINE LEARNING RESEARCHER 04/2011 – 05/2014 Work closely with development teams to ensure accurate integration of machine learning models into firm platforms This role will suit you if you thrive on working in a fast paced environment where your work has high impact Develop the team's capabilities in data science and machine-learning, and apply them to create new data-driven insights Create innovative, systematic investment signals and strategies based on a rigorous, peer-reviewed research process Work with every investment team in MBFI (Credit, Macro, Equities, RV, Securitized, Mid-horizon etc.) to introduce cutting-edge techniques and innovative data sources across the Fund Design, conduct, and report results from prototype or proof-of-concept research projects that focus on 1) new tools, methods, or algorithms, 2) new scientific domains or application areas, or 3) new data sets or sources Develop new machine learning models to detect malicious activity on mobile devices MACHINE LEARNING 10/2006 – 10/2010 Research & develop Machine Learning models for security problems, in the areas of Networking, Application & Data Provide SW specifications and production quality code on time to meet project mile stones Provide SW specifications, production quality code and engage with algorithm proliferation activities Research and develop deep learning algorithms Research and develop state of the art techniques in the field of computer vision, ML and DL Work with the system, physics, SW and application group Develop the next generation of automation tools for monitoring and measuring Ad Quality, with associated user interfaces EDUCATION Bachelor's Degree in Computer Science JOHNSON & WALES UNIVERSITY SKILLS Excellent organizational and analytical skills Experience of developing data science capability Excellent written and oral communication skills Ability to read, understand, and communicate technical documentation Strong communication, presentation and business and technical writing skills Strong background in Python, SQL, and R. Familiarity with designing and conducting experiments with human subjects Share knowledge with people known for their world-class expertise and insight Relevant technical and delivery experience within, or working as a consultant/advisor to, a financial services organisation Work with global clients, locally and internationally Be exposed to different business cultures in high-performing teams"

def compare_job_skills(skills_list):
    skills_tool_list = []
    skills_skill_list = []
    skills_degree_list = []

    return_skills = []

    match_skills = prepare_job_desc(skills_list)
    skills_desc = skills_list.lower()
    skills_desc_set = match_skills

    # check if the keywords are in the job description. Look for exact match by token.
    skills_tool_words = tool_keywords1_set.intersection(skills_desc_set)
    skills_skill_words = skill_keywords1_set.intersection(skills_desc_set)
    skills_degree_words = degree_keywords1_set.intersection(skills_desc_set)

    # check if longer keywords (more than one word) are in the job description. Match by substring.
    j = 0
    for tool_keyword2 in tool_keywords2:
        # tool keywords.
        if tool_keyword2 in skills_desc:
            skills_tool_list.append(tool_keyword2)
            j += 1
    
    k = 0
    for skill_keyword2 in skill_keywords2:
        # skill keywords.
        if skill_keyword2 in skills_desc:
            skills_skill_list.append(skill_keyword2)
            k += 1

    # search for the minimum education.
    min_education_level = 999
    for degree_word in degree_words:
        level = degree_dict[degree_keywords1_dict[degree_word]]
        min_education_level = min(min_education_level, level)
    
    
    for degree_keyword2 in degree_keywords2:
        # longer keywords. Match by substring.
        if degree_keyword2 in skills_desc:
            level = degree_dict2[degree_keyword2]
            min_education_level = min(min_education_level, level)
    
    # label the job descriptions without any tool keywords.
    if len(skills_tool_words) == 0 and j == 0:
        skills_tool_list.append('nothing specified')
    
    # label the job descriptions without any skill keywords.
    if len(skills_skill_words) == 0 and k == 0:
        skills_skill_list.append('nothing specified')
    
    # If none of the keywords were found, but the word degree is present, then assume it's a bachelors level.
    if min_education_level > 500:
        if 'degree' in skills_desc:
            min_education_level = 1
    
    skills_tool_list += list(skills_tool_words)
    skills_skill_list += list(skills_skill_words)
    #skills_degree_list.append(min_education_level)

    return_skills += list(skills_tool_list)
    return_skills += list(skills_skill_list)
    #return_skills += list(skills_degree_list)   
 
    return return_skills
    #match_skills_dict = {ps.stem(tok):tok for tok in match_skills}
"""     print(df_new['keyword'])
    for skill in match_skills:
        print(skill)
        if skill in df_new['keyword'].values:
            #if a['Names'].str.contains('Mel').any():
            print(tool_keywords1_dict.get(skill))
        #[number_map[int(x)] for x in input_str.split()]
    #tool_keywords1_dict
    return match_skills
 """


final_skills = compare_job_skills(skills_list)
print(final_skills)