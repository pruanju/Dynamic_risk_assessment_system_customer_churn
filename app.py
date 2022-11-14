from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filepath = request.get_json()['filepath']

    X = pd.read_csv(filepath)
    predictions = model_predictions(X)

    return jsonify(predictions.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats1():        
    #check the score of the deployed model
    score_model()
    with open(os.getcwd()+'/'+model_path+'/'+'latestscore.txt', 'r') as f:
        f1_score = f.read()

    return f1_score

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats2():        
    summary = dataframe_summary()
    return  jsonify(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats3():        
    #check timing and percent NA values
    num_nulls = missing_data() 
    timing = execution_time() 
    outdated_packages = outdated_packages_list()

    all = {
        'missing_percentage': num_nulls,
        'execution_time': timing,
        'outdated_packages': outdated_packages
    }

    return jsonify(all)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
