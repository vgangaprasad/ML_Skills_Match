#load the data
import csv as csv
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import pos_tag
from models.keywords import tool_keywords1
from models.keywords import tool_keywords2
from models.keywords import skill_keywords1
from models.keywords import skill_keywords2
from models.keywords import degree_dict
from models.keywords import degree_dict2
from models.keywords import degree_keywords2
class ResumeToSkill(object):
    def __init__(self):
        #df = pd.read_csv("csv_ML.csv")
        global ps
        

        # skills_list = "EXPERIENCE Los Angeles, CA SOFTWARE ENGINEER, MS, MACHINE LEARNING 08/2014 – present Work with Data Scientists and Product Managers to frame a problem, both mathematically and within the business context Deploy validated algorithms to our RTB system, and develop techniques for monitoring and visualizing performance of all deployed algorithms Knowledge developing and debugging in C/C++ and Java Knowledge of one or more open-source Machine Learning framework Develop prototypes and validate the results Expert knowledge developing and debugging in C/C++ and Java Contribute to the production solutions' development, testing and deployment Phoenix, AZ MACHINE LEARNING RESEARCHER 04/2011 – 05/2014 Work closely with development teams to ensure accurate integration of machine learning models into firm platforms This role will suit you if you thrive on working in a fast paced environment where your work has high impact Develop the team's capabilities in data science and machine-learning, and apply them to create new data-driven insights Create innovative, systematic investment signals and strategies based on a rigorous, peer-reviewed research process Work with every investment team in MBFI (Credit, Macro, Equities, RV, Securitized, Mid-horizon etc.) to introduce cutting-edge techniques and innovative data sources across the Fund Design, conduct, and report results from prototype or proof-of-concept research projects that focus on 1) new tools, methods, or algorithms, 2) new scientific domains or application areas, or 3) new data sets or sources Develop new machine learning models to detect malicious activity on mobile devices MACHINE LEARNING 10/2006 – 10/2010 Research & develop Machine Learning models for security problems, in the areas of Networking, Application & Data Provide SW specifications and production quality code on time to meet project mile stones Provide SW specifications, production quality code and engage with algorithm proliferation activities Research and develop deep learning algorithms Research and develop state of the art techniques in the field of computer vision, ML and DL Work with the system, physics, SW and application group Develop the next generation of automation tools for monitoring and measuring Ad Quality, with associated user interfaces EDUCATION Bachelor's Degree in Computer Science JOHNSON & WALES UNIVERSITY SKILLS Excellent organizational and analytical skills Experience of developing data science capability Excellent written and oral communication skills Ability to read, understand, and communicate technical documentation Strong communication, presentation and business and technical writing skills Strong background in Python, SQL, and R. Familiarity with designing and conducting experiments with human subjects Share knowledge with people known for their world-class expertise and insight Relevant technical and delivery experience within, or working as a consultant/advisor to, a financial services organisation Work with global clients, locally and internationally Be exposed to different business cultures in high-performing teams"
        
    # process the job description and return filtered tokens
    # def prepare_job_desc(self, desc):
        

    #skills_list = "Statistics bayesian Machine Learning python Visualization"
    #this function takes skills_list as raw text input and calls prepare_job_desc preprocessing then returns the skills after parsing for keywords as a python list
    def compare_job_skills(self, skills_list):
        ps = PorterStemmer()

        tool_keywords1_set = set([tok for tok in tool_keywords1]) # stem the keywords (since the job description is also stemmed.)
        tool_keywords1_dict = {tok:tok for tok in tool_keywords1} # use this dictionary to revert the stemmed words back to the original.

        skill_keywords1_set = set([tok for tok in skill_keywords1])
        skill_keywords1_dict = {tok:tok for tok in skill_keywords1}

        degree_keywords1_set = set([tok for tok in degree_dict.keys()])
        degree_keywords1_dict = {tok:tok for tok in degree_dict.keys()}

        # tokenize description.
        # skills_list = "EXPERIENCE Los Angeles, CA SOFTWARE ENGINEER, MS, MACHINE LEARNING 08/2014 – present Work with Data Scientists and Product Managers to frame a problem, both mathematically and within the business context Deploy validated algorithms to our RTB system, and develop techniques for monitoring and visualizing performance of all deployed algorithms Knowledge developing and debugging in C/C++ and Java Knowledge of one or more open-source Machine Learning framework Develop prototypes and validate the results Expert knowledge developing and debugging in C/C++ and Java Contribute to the production solutions' development, testing and deployment Phoenix, AZ MACHINE LEARNING RESEARCHER 04/2011 – 05/2014 Work closely with development teams to ensure accurate integration of machine learning models into firm platforms This role will suit you if you thrive on working in a fast paced environment where your work has high impact Develop the team's capabilities in data science and machine-learning, and apply them to create new data-driven insights Create innovative, systematic investment signals and strategies based on a rigorous, peer-reviewed research process Work with every investment team in MBFI (Credit, Macro, Equities, RV, Securitized, Mid-horizon etc.) to introduce cutting-edge techniques and innovative data sources across the Fund Design, conduct, and report results from prototype or proof-of-concept research projects that focus on 1) new tools, methods, or algorithms, 2) new scientific domains or application areas, or 3) new data sets or sources Develop new machine learning models to detect malicious activity on mobile devices MACHINE LEARNING 10/2006 – 10/2010 Research & develop Machine Learning models for security problems, in the areas of Networking, Application & Data Provide SW specifications and production quality code on time to meet project mile stones Provide SW specifications, production quality code and engage with algorithm proliferation activities Research and develop deep learning algorithms Research and develop state of the art techniques in the field of computer vision, ML and DL Work with the system, physics, SW and application group Develop the next generation of automation tools for monitoring and measuring Ad Quality, with associated user interfaces EDUCATION Bachelor's Degree in Computer Science JOHNSON & WALES UNIVERSITY SKILLS Excellent organizational and analytical skills Experience of developing data science capability Excellent written and oral communication skills Ability to read, understand, and communicate technical documentation Strong communication, presentation and business and technical writing skills Strong background in Python, SQL, and R. Familiarity with designing and conducting experiments with human subjects Share knowledge with people known for their world-class expertise and insight Relevant technical and delivery experience within, or working as a consultant/advisor to, a financial services organisation Work with global clients, locally and internationally Be exposed to different business cultures in high-performing teams"
        tokens = word_tokenize(skills_list)
            
        # Parts of speech (POS) tag tokens.
        token_tag = pos_tag(tokens)
        
        # Only include some of the POS tags.
        include_tags = ['VBN', 'VBD', 'JJ', 'JJS', 'JJR', 'CD', 'NN', 'NNS', 'NNP', 'NNPS']
        filtered_tokens = [tok.lower() for tok, tag in token_tag if tag in include_tags]
        
        # stem words.
        stemmed_tokens = [ps.stem(tok).lower() for tok in filtered_tokens]
        #return set(stemmed_tokens)
        match_skills = set(filtered_tokens)
        skills_tool_list = []
        skills_skill_list = []
        skills_degree_list = []

        return_skills = []

        # match_skills = prepare_job_desc(skills_list)
        skills_desc = skills_list.lower()
        skills_desc_set = match_skills

        #print(skills_desc_set)
        # check if the keywords are in the job description. Look for exact match by token.
        skills_tool_words = tool_keywords1_set.intersection(skills_desc_set)
        skills_skill_words = skill_keywords1_set.intersection(skills_desc_set)
        skills_degree_words = degree_keywords1_set.intersection(skills_desc_set)
        #print(degree_keywords1_set)

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
        for degree_word in skills_degree_words:
            level = degree_dict[degree_keywords1_dict[degree_word]]
            min_education_level = min(min_education_level, level)
        
        
        for degree_keyword2 in degree_keywords2:
            # longer keywords. Match by substring.
            if degree_keyword2 in skills_desc:
                level = degree_dict2[degree_keyword2]
                min_education_level = min(min_education_level, level)
        
        # label the job descriptions without any tool keywords.
        # if len(skills_tool_words) == 0 and j == 0:
        #     skills_tool_list.append('nothing specified')
        
        # # label the job descriptions without any skill keywords.
        # if len(skills_skill_words) == 0 and k == 0:
        #     skills_skill_list.append('nothing specified')
        
        # If none of the keywords were found, but the word degree is present, then assume it's a bachelors level.
        if min_education_level > 500:
            if 'degree' in skills_desc:
                min_education_level = 1
        
        skills_tool_list += list(skills_tool_words)
        skills_skill_list += list(skills_skill_words)
        #skills_degree_list.append(min_education_level)

        return_skills += list(skills_tool_list)
        return_skills += list(skills_skill_list)
        return_skills += list(skills_degree_list)   
    
        return return_skills


# final_skills = compare_job_skills(skills_list)
# print(final_skills)