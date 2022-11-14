import pandas as pd
import numpy as np
import os
import logging
import json
import ast
from sklearn import metrics
from diagnostics import model_predictions


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 


input_folder_path = config['input_folder_path']
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']

##################Check and read new data
#first, read ingestedfiles.txt
# Using readlines()

logging.info("Checking for new data")
with open(os.getcwd()+'/'+prod_deployment_path+'/'+'ingestedfiles.txt', 'r') as f:
    files = f.readlines()

  
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = [x for x in os.listdir(os.getcwd()+'/'+input_folder_path+'/') if x[-4:]=='.csv']
for new_file in new_files: 
    if new_file in files:
        new_files.remove(new_file) 

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) > 0:
    logging.info("Ingesting new data")
    os.system('python3 ingestion.py') 
else:
    logging.info("No new data found")
    exit()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
logging.info("Checking for model drift")
with open(os.getcwd()+'/'+prod_deployment_path+'/'+'latestscore.txt', 'r') as f:
    deployed_score = ast.literal_eval(f.read())

new_df = pd.read_csv(os.getcwd()+'/'+output_folder_path+'/'+'finaldata.csv')
new_predictions = model_predictions(new_df)
new_real =  new_df['exited'].to_numpy()
new_f1_score = metrics.f1_score(new_predictions, new_real)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if(new_f1_score >= deployed_score):
    logging.info("No model drift occurred")
    exit()
else:
    logging.info("Model drift occurred")
    # Re-training the model
    logging.info("Training again the model with new data")
    os.system('python3 training.py')
    # After training the system we update the score of the new model
    os.system('python3 score.py')


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
logging.info("Deploying to production")
os.system('python3 deployment.py')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
logging.info("Reporting & API calls")
os.system('python3 reporting.py')
os.system('python2 apicalls.py')






