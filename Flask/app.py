import os
from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory, current_app, jsonify
#from models.keras_first_go import KerasFirstGoModel
from models.skill_to_job import SkillToJob
from models.main import ResumeToSkill
# import skill_to_job
import simplejson as json

#################################################
# Flask Setup
#################################################


#################
# Model config
#################
app = Flask(__name__, static_folder='static', static_url_path='')



def train_model():
    global first_go_model
    global skill_to_job
    global resume_to_skill 
    
    print("Train the model")
    # first_go_model = KerasFirstGoModel()
    skill_to_job = SkillToJob()
    #create instances of each class from the model
    resume_to_skill = ResumeToSkill()
    

@app.route('/<path:path>')
def serve_page(path):
    return send_from_directory('client', path)

@app.route("/")
def welcome():
    return render_template('index.html')

@app.route('/', methods=['Get', 'POST'])
def my_form_post():
    text = request.form.get('job1')
    processed_text = text.upper()
    print (processed_text)
    return render_template('result.html', **templateData)

# @app.route('/api/jobs', methods=['Get', 'POST'])
# def get_jobs():


@app.route('/submitted', methods=['POST', 'GET'])
def handle_data():
    # Retreive the form text using the key 'job' which is the form id
    # retrieve the form only if a request is made
    if request.method == "POST":
        result_1 = request.form['job1']
        
        train_model()
        results = resume_to_skill.compare_job_skills(result_1)
        print (results)
        # results is an array of strings of some length, but skill to job requires one big string
        # so we convert results back to a big string via .join 
        skills_string = " ".join(results)
        jobs = skill_to_job.prediction(skills_string)
    return render_template('index.html', results=results, jobs=jobs)   



if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')