from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def copy_files():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    latest_score_file_path = os.getcwd()+'/'+model_path+'/'+'latestscore.txt'
    ingested_file_path = os.getcwd()+'/'+output_folder_path+'/'+'ingestedfiles.txt'
    model_file_path = os.getcwd()+'/'+model_path+'/'+'trainedmodel.pkl'
    os.system(f"cp -p {latest_score_file_path} {prod_deployment_path}")
    os.system(f"cp -p {ingested_file_path} {prod_deployment_path}")
    os.system(f"cp -p {model_file_path} {prod_deployment_path}")
        
        
        
if __name__ == '__main__':
    copy_files()
