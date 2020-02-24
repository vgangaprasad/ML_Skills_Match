import json as j
import csv as csv
import sys
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math
import string
import os


csv_data = None
rows = []
with open('./Resources/clean_data.csv') as data_file:
    csv_data = csv.reader(data_file, delimiter=',', quotechar='|')
    for row in csv_data:
        rows.append(row)

data = pd.read_csv("./Resources/clean_data.csv")
# data.head()
# print(data.head())
print(data['description'],data['label'])


""" data = pd.DataFrame(json_data) """

stop_words=' '.join(list(data[data['label']=='negative']['description']))
stop_wc= WordCloud(width=512,height=512).generate(stop_words) 
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(stop_wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

def get_data():
    data = pd.read_csv("./Resources/clean_data.csv")
    # data.head()
    data.loc[(data['label'] == 'positive'),'label']=1
    data.loc[(data['label'] == 'negative'),'label']=0
    return data['description'],data['label']

class nonMLDetector(object):
    """Implementation of Naive Bayes for binary classification"""
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)
 
    def tokenize(self, text):
        # text = self.clean(text).lower()
        # words = stopwords.words("english")
        # return re.split("\W+", text)
        stop_words = set(stopwords.words('english')) 
        stop_list=['<', 'p', '>', 'We', 'hiring', 'work', 'across', 'entire', 'technology', '.', 'Our', 'exciting', 'products', ',', 'advanced', 'matching', 'leveraging', 'tools', 'guide', 'candidates', 'first', 'job', 'company', 'building', 'solutions', 'amazing', 'end', '&', 'nbsp', ';', '/p', 'br', '/', 'expected', 'product', 'influence', 'decisions', 'constantly', 'looking', 'new', 'ways', 'improve', 'engineering', 'team', 'quickly', 'This', 'position', 'reports', 'directly', 'strong', 'What', ':', '/strong', 'ul', 'li', 'Deep', 'curiosity', '/li', 'A', 'high', 'excellent', 'user', 'experience', 'Strong', 'skills', 'Ability', 'write', 'clean', 'code', '/ul', 'Preferred', 'Qualifications', '2+', 'years', 'preferably', 'startup', 'Previous','modern', '(', 'Vue',')', 'At', 'href=', "''", 'http', 'rel=', 'nofollow', '/a', 'believe', 'everyone', 'rsquo', 'community', 'employees', 'ensure', 'someone', 'beyond', 'helping', 'million', 'people', 'us', 'journey', 'site', 'mobile', 'apps', 'love', 'join', 'change', 'lives', 'You', 'tech', 'make', 'also', 'happen', 'help', 'one', 'meaningful', 'things', 'US', 'including', 'great', 'jobs', 'easier', 'difference', 'together', 'For', 'Build', 'full', 'applications', 'features', 'using', 'Native', 'around', 'Design', 'developer', 'support', 'infrastructure', 'Experience', 'collaborative', 'environment', 'leading', 'front', 'operating', 'critical', 'systems', 'technical', 'use', 'Docker', 'broad', 'exposure','An', 'mindset', 'always', 'grow', 'ability', 'date', 'getting', 'writing', 'B.S', 'Computer', 'Science', 'related', 'field', 'equivalent', 'professional', 'em', 'important', '/em', 'Equal', 'Opportunity', 'age', 'color', 'national', 'origin', 'race', 'religion', 'gender', 'sex', 'sexual', 'orientation', 'identity', 'and/or', 'expression', 'marital', 'status', 'veteran', 'characteristic', 'protected', 'federal', 'state', 'local', 'In', 'addition', 'provide', 'reasonable', 'accommodations', 'qualified', 'As', 'part', 'lead', 'engineer', 'build', 'developers', 'take', 'next', 'level', 'client', 'right', 'advance', 'vision', 'candidate', 'efforts', 'ecosystem', 'leaders', 'teams', 'see', 'big', 'ready', 'commit', 'engage', 'day', 'growing','Responsibilities', 'Contribute', 'highly', 'activities', 'multiple', 'internal', 'customer', 'regular', 'senior', 'control', 'Own','project', 'success', 'measure', 
        'report', 'Be', 'find','organizational', 'challenges', 'Write', 'test', 'plans', 'extraordinary', 'attention', 'detail', 'Required', 'Skills', 'deep', 'expertise', 'performance', 'tuning', 'working', 'technologies','plus', 'e.g', 'etc', 'Exposure','language', 'web', 'financial', 'emphasis', 'preferred', 'Excellent', 'set', 'appropriate', 'achieve', 'successful', 'results', 'The', 'Baseball', 'talented', 'computer', 'Research', 'amp', 'Development', 'within', 'Operations', 'plan', 'design', 'collaborate', 'player', 'acquisition', 'video', 'seeking', 'knowledge', 'latest', 'novel', 'u', 'Duties', '/u', 'Collaborate','maintaining','Assist','value', 'Help', 'deliver', 'future', 'quantitative', 'processing', 'baseball', 'game', 'training', 'etc.', 'modeling', 'Proficiency','degree', 'camera', 'hardware', 'libraries','automated', 'iPhone', 'world', 'like', 'experiences', 'Camera', 'image', 'ship', 'millions', 'want', 'Are', 'contribute', 'digital', '?', 'Apples', 'Software', 'passionate', "'s", 'span', 'develop','If', 'problems', 'focus', 'every', 'real', 'you.', '/span', 'Come', 'rapidly', 'family', 'products.', 'networks', 'place', 'Senior', 'Engineer', 'performing', 'experienced', 'able', 'independently', 'capable', 'high-level', 'requirements', 'following', 'essential', 'role', 'time', 'application','similar', 'offer', 'competitive', 'benefits', 'encourage', 'ideas', 'require', 'WE', 'makes', 'proud', 'back', 'energy', 'transformation', 'values', 'professionals', '10', 'top', 'many', 'largest', 'companies', 'customers', 'market', 'reach', 'impact', 'solve', 'industry', 'problems.', 'Go', 'solution', 'Join','creativity', 'Work', 'stability', 'base', 'growth', 'offers', 'compensation', 'full-time', 'medical/dental/vision', 'paid', 'vacation', 'holidays', '401', 'k', 'YOU', 'LL', 'industry-leading', 'real-time', 'scientific', 'much', 'ground', 'There', 'designs', 'builds', 'major', 'components', 'approaches', 'end-to-end', '#', 'empowered', 'business', 'RESPONSIBILITIES', 'high-quality', 'ownership', 'unit', 'tests', 'members', 'define', 'release', 'monitoring', 'functionality', 'service', 'production', 'resume', 'Please', 'include', 'interest', 'space', 'large', 'number', 'applicants', 'positions', 'meet', 'located', 'Founded', '2015', 'Even', 'Financial', 'transforming', 'way', 'institutions', 'consumers', 'By', 'seamlessly', 'connecting', 'American', 'Express', 'Goldman', 'Sachs', 'channel', 'partners', 'Smart', 'via', 'consumer', 'programmatic', 'source', 'compliance','loans','backed', 'Ventures', 'Capital', 'strives', 'foster', 'functions', 'levels', 'personal', 'mentality', 'potential', '-', 'wait', 'empirical', 'explore', 'iterate', 'transparent', 'helpful', 'straightforward', 'honest', 'fearless', 'approach', 'feedback', 'process.', 'document', 'inside', 'questions', 'move', 'Must', 'experience.', 'least','component', 'visual', 'understand', 'good', 'Nice', 'code.', 'TypeScript', 'Implement', 'rendering', 'Full', 'eligible', 'comprehensive', 'package', 'provided', 'Blue', 'Life', 'disability', 'insurance', 'contributions', 'time.', 'Stock', 'option', 'program.', 'Paid', 'PTO', 'accordance', 'Company', 'policy', 'New', 'Hire', 'gym', 'membership', 'mission', 'NSA', '``', 'different', 'organization', 'Scientists', 'well', 'requires', 'threats', 'global', 'Support', 'share', 'contributing', 'open', 'world-class', 'must', 'drive', 'artificial', 'computing', 'IT', 'create', 'available', 'environments', 'research', 'may', 'considered', 'program', 'opportunity', 'receive', 'Agency', 'career', 'Scientist', 'responsibilities', 'creating', 'sets', 'sources', 'integrating', 'developed','Developer', 'identifying', 'developing', 'capabilities', 'enable', 'collection', 'Internet', 'analyzing', 'identify', 'ideal', 'solving', 'handle', 'information', 'apply', 'structures', 'operate', 'needs', 'validate', 'e.g.', 'storage', 'administration', 'retrieval', 'statistics','device','Benefits', 'courses', 'external', 'based', 'need', 'basic', 'flexible', 'Summary', 'unique', 'Do', 'others', 'culture', 'yet', 'protect', 'challenge', '!', 'Salary', 'Range', '$', '*The', 'qualifications', 'listed', 'minimum', 'acceptable', "'", 'education', 'relevant', 'account', 'manager/organization', 'regarding', 'position.', 
        'Entry', 'Bachelor', 'Associate', '2', 'individuals', 'in-depth', 'clearly', 'Degree', 'CS', 'Related', 'fields', 'Engineering', 'programs', 'contain', 'concentration', 'foundational', 'areas', 'languages','mathematics', 'example', 'discrete', 'Information', 'Technology', 'Systems', 'IS', 'degrees', 'amount', 'type', 'coursework', 'major.', 'Relevant', 'process', 'i.e.', 'deployment/installation', 'maintenance', 'programming.', '3', 'Master', '1', 'year', 'Doctoral', '5', 'To', 'please', 'email', '@', 'applying', 'Security', 'Digital', 'Data', 'required', 'disabilities', 'equal', 'employer', 'applicable', 'employment', 'laws', 'All', 'subject', 'drug', 'upon', 'background', 'Intelligence', 'System', 'DCIPS', 'DoD', 'Title', 'procedures', 'asked', 'documents', 'Your', 'used', 'providing', 'result', 'Job', 'could', 'due', 'States', 'Interactions', 'worlds','two', 'todays', 'industries', 'operational', 'accelerate', 'conversational', 'communicate', 'tasks', 'without', 'benefit', 'play', 'key', 'b', '/b', 'Develop', 'feature', 'delivery','field.', 'hands-on', 'solid', 'understanding', 'virtual', 'leverage', 'best', 'challenging', 'team.', 'Prior', 'statistical', 'methods', 'dedicated', 'Whether', 'health', 'pay', '401k', 'leave', 'committed', 'individual', 'discriminate', 'basis', 'pregnancy', 'law.', 'projects', 'clients.', 'THE', 'diverse', 'Architect', 'delivering', 'clients', 'ndash', 'alongside', 'groups', 'responsibility', 'cross', 'functional', 'leads', 'risk', 'aspects', 'Health', 'goal', 'bring', 'get', 'mentor', 'effective', 'practices', 'foundation', 'relationships','insights', 'extensive', 'non-technical', 'environment.','Hands-on', 'variety','Infrastructure', 'native', 'reduce', 'small', 'fast', 'know', 'deploying', 'providers','Google', 'patterns', 'driven', 'continuous', 'system', 'wide', 'range', 'structured', 'salary', 'enabling', 'development.', 'care', 'clear', 'give', 'making', 'fun', 'brands', '100', 'ldquo', 'rdquo', 'gives','opportunities', 'social', 'balance', 'travel', 'various', 'participate', 'proven', 'necessary', 'Applicant', 'Have', 'life', 'cycle', 'enterprise', 'specifications','plus.', 'interact', 'resolve', 'requests', 'effectively', 'techniques', 'Has', 'platforms', 'trends', 'healthcare', 'About', 'advertising', 'execute', 'media', 'TV',
         'marketing', 'API.','content', 'owners', 'audiences', 'scientists', 'evolve', 'billion', 'member', 'planning', 'scaling', 'existing', 'teams.', 'managing','technologies.', 'Able', 'CA', 'And', 'come', 'snacks', 'platform', 'learn', 'helps', 'better', 'Create', 'devices', 'Augmented', 'Reality', '3D', 'Requirements', '3+', 'group', 'issues', 'Demonstrated', 'APIs', 'interested', 'https', 'USA', 'View', 'Minimum', 'practical', 'purpose', 'languages.', 'Working', 'limited', 'diversity', 'services.', 'next-generation', 'billions', 'users', 'massive', 'search', "'re", 'large-scale', 'fast-paced', 'push', 'deploy', 'maintain', 'Mira', 'primary', 'implement', 'Unity', 'Evaluate', 'optimize','consistent', 'optimization','Familiarity','concepts', 'operations', 'Bonus', 'Knowledge', 'With', '50', 'valuable', 'shape', 'built', 'integrity', '5+', 'mind', 'Good', 'common', 'packages', '%', 'Flexible', 'Team', 'insight', 'curious', 'problem', 'generation', 'positive', 'impacts', 'strategies', '7+', 'huge', 'skills.', 'Action', 'class', 'closely','events', 'fit', 'data-driven', 'current', 'responsible', 'engaging', 'Drive', 'efficiency', 'utilizing', 'strategic', 'recommendations', 'database', 'decision', 'ensuring', 'driving', 'Quality', 'goals', 'monitor', 'even', "'ll", 'Apple', 'consider', 'Were', 'Key','[', ']', 'platform.', 'Indeed','daily', '*', 'pace', 'Cloud', 'long-term', 'solutions.', 'provides', 'Embedded', 'Lead', 'M.S', 'conditions', 'center', 'medical', 'generous', 'office', '4+', 'record', 'standard', 'Rockstar', 'would', 'edge', 'heart', 'downtown', 'collaboration', 'empower', 'analysts', 'applications.',
        'efficient', 'scale.','app','peer','reviews','Lambda','data.','pride','equity','Four','weeks','cloud.','Dental','Vision','Policy','401K','Term','Disability','One','Medical','Unlimited','individuals.','continuously','Altitude','paying','components.','stack','framework','hard','works','end', 'ad','web','oriented','role','exceptional','use','interface','core','day','track','forward', 'demonstrated', 'innovation', 'Tech', 'leader', 'possible', 'excited', 'It', 'smart', 'enjoy', 'think', 'power', 'access', 'hundreds', 'organizations', 'portfolio', 'Big', 'discovery','dental', 'action', 'Learn', 'keep', 'let', 'efforts.', 'Mac', 'Recursion', 'thousands', 'biology', 'transgender', 'physical', 'citizenship', 'Facebook', 'closer', 'Through', 'kind', 'connects', 'matters', 'expand', 'builders', 'iterating', 'Together', 'stronger', 'communities', 'started.', 'Want', 'Engineers', 'per', 'globe', 'office.','Affirmative', 'childbirth', 'stereotypes', 'legally', 'characteristics', 'criminal', 'histories', 'recruiting', 'assistance', 'accommodations-ext', 'fb.com.', 'systems.', 'AR', 'Facebooks', 'expert', 'Communicate', 'capture', 'Simon', 'businesses', 'attitude', 'partner', 'currently', 'staff', 'Oregon', 'call', 'Understanding', 'Product', 'Managers', 'drivers', 'Metal', 'Building', 'On', 'VR', 'money','Givelify', 'causes', 'CI/CD','environments.', 'bull', 'initiatives', 'evaluate', 'inspire','Illustrator', 'offerings','IAS', 'Earnin', 'Glassdoor', 'Database', 'browser', 'Response', 'Center',]
        word_tokens = word_tokenize(text) 
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)
        clean_sentence = [w for w in filtered_sentence if not w in stop_list]
        return clean_sentence
 
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        # print('Word_Count',str(word_counts))
        return word_counts

    def fit(self, X, Y):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()
    
        n = len(X)
        self.num_messages['nonML'] = sum(1 for label in Y if label == 0)
        self.num_messages['ML'] = sum(1 for label in Y if label == 1)
        self.log_class_priors['nonML'] = math.log(self.num_messages['nonML'] / n)
        self.log_class_priors['ML'] = math.log(self.num_messages['ML'] / n)
        self.word_counts['nonML'] = {}
        self.word_counts['ML'] = {}
    
        for x, y in zip(X, Y):
            c = 'nonML' if y == 'negative' else 'ML'
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                # print(counts.items())

                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
    
                self.word_counts[c][word] += count

        newList = []
        # Iterate over all the items in dictionary and filter items based on values
        for (key, value) in self.word_counts['ML'].items():
        # Check value and then add key to new dictionary
            if value >= 5.0:
                newList.append(key) 
        additional_word_list=['scala','postgres','css','API','javascript','js','net','unix','architechture','sql','linux','motivated','transform']
        for i in range(len(additional_word_list)):
            newList.append (additional_word_list[i])   
        # print('Filtered Dictionary : ')
        # print(newList)

        # Create word cloud for ML related terms
        ml_words=' '.join(newList)
        ml_wc= WordCloud(width=512,height=512).generate(ml_words) 
        plt.figure(figsize=(10,8),facecolor='k')
        plt.imshow(ml_wc)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

        with open("./Resources/bag_of_words_jobProfile.csv","w") as f:
            wr = csv.writer(f,delimiter="\n")
            wr.writerow(newList)

    def predict(self, X):
        result = []
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            nonML_score = 0
            ml_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue
                
                # add Laplace smoothing
                log_w_given_nonML = math.log( (self.word_counts['nonML'].get(word, 0.0) + 1) / (self.num_messages['nonML'] + len(self.vocab)) )
                log_w_given_ml = math.log( (self.word_counts['ML'].get(word, 0.0) + 1) / (self.num_messages['ML'] + len(self.vocab)) )
    
                nonML_score += log_w_given_nonML
                ml_score += log_w_given_ml
    
            nonML_score += self.log_class_priors['nonML']
            ml_score += self.log_class_priors['ML']
            if nonML_score < ml_score:
                result.append(1)
            else:
                result.append(0)
        return result

if __name__ == '__main__':
    X, y = get_data()
    MNB = nonMLDetector()
    MNB.fit(X[200:], y[200:])
 
    pred = MNB.predict(X[:200])
    true = y[:200]
     
    accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))

    sample_profile="Duties include, but are not limited to: Knowledge of capabilities and intricacies of information systems. Ability to understand and carry out moderately complex technical instructions and requests. Ability to work effectively with users, vendors, and other IT personnel. Excellent verbal and written communication skills; energetic, self-motivated, multi-tasking, team player with excellent organizational and administrative abilities; excellent interpersonal skills and ability to work with a diverse group of professionals in a demanding environment; advanced knowledge of computer software programs with advanced mathematical and analytical skills; strong time management skills with strict attention to detail to establish priorities, manage multiple projects, evaluate outcomes, and drive to deadlines."
    label='negative'
    sample_obj=nonMLDetector()
    sample_obj.fit(sample_profile,label)
    sample_pred=sample_obj.predict(sample_profile)
    print('sample_pred', sample_pred)

    


