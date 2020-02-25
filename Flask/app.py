import os
from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory, current_app, jsonify
from models.keras_first_go import KerasFirstGoModel
from models.skill_to_job import SkillToJob
# import skill_to_job
from clear_bash import clear_bash
import simplejson as json

#################################################
# Flask Setup
#################################################


#################
# Model config
#################
app = Flask(__name__, static_folder='static', static_url_path='')


# cleaner=clear_bash()

def train_model():
    global first_go_model
    global skill_to_job

    print("Train the model")
    first_go_model = KerasFirstGoModel()
    skill_to_job = SkillToJob()
    

@app.route('/<path:path>')
def serve_page(path):
    return send_from_directory('client', path)

@app.route("/")
def welcome():
    return render_template('index.html')

@app.route('/', methods=['Get', 'POST'])
def my_form_post():
    text = request.form.get('job')
    processed_text = text.upper()
    print (processed_text)
    return render_template('result.html', **templateData)

@app.route('/submitted', methods=['POST', 'GET'])
def handle_data():
    # Retreive the form text using the key 'job' which is the form id
    result_1 = request.form['job1']
    result_2 = request.form['job2']
    result_3 = request.form['job3']
    result_4 = request.form['job4']
    train_model()
    list_prediction = skill_to_job.prediction(result_1)
    print(list_prediction)
    # preict by passing in form data
    processed_text = first_go_model.prediction(result)
    print(processed_text)
    result = {'Job': processed_text}
    # data = json.dumps([dict(r) for r in result])
    return render_template('index.html', result=processed_text[0])   


# @app.route('/result',methods = ['POST', 'GET'])
# def result():
#    if request.method == 'POST':
#       result = request.form.getlist('Job')
#       train_model()
#       processed_text = first_go_model.prediction(result[0])
#       result = {'Job': processed_text}
#     #   return render_template("result.html",result = result)
#      return json.dumps([dict(r) for r in result])

def clear_bash():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    clear_bash()
    print("---------------------------------")
    print("JOB PREDICTION APPLICATION")
    print("---------------------------------")
    print("\n\n")
    print("--- SET MODE ---")
    print("\n")
    print("Do you want to start the application or test the model? ")
    print("\n")
    print("NOTE: Press 'app' for application of 'model' for model...")
    mode = input()

    if mode=='app':

        clear_bash()

        print("---------------------------------")
        print("JOB PREDICTION APPLICATION")
        print("---------------------------------")
        print(("*Flask starting server..."
                   "please wait until server has fully started"))
        app.debug = True
        app.run(use_reloader=False)

    elif mode=='model':

        clear_bash()

        print("---------------------------------")
        print("PLAY WITH KERAS MODEL")
        print("---------------------------------")

        print("Wait until I'm ready..")
        train_model()
        print("\n\n")
        print("OK the model is ready!!")
        skills_description = input("Give me your skill description please: ")
        prediction = first_go_model.prediction(skills_description)
        clear_bash()
        print("This job is my suggetion: ", prediction)
        print("Thanks for all!!")
        clear_bash()

    else:
        print("You are stupid!! BYEEE")
