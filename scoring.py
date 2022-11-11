from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(os.getcwd()+'/'+model_path+'/'+'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    test_data = pd.read_csv(os.getcwd()+'/'+test_data_path+'/'+'testdata.csv')
    
    y_real = test_data['exited'].to_numpy()
    X = test_data.drop(columns=['corporation', 'exited'], axis=1).to_numpy()

    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y_pred, y_real)

    with open(os.getcwd()+'/'+model_path+'/'+'latestscore.txt', 'w+') as f:
        # Write elements of the list of dataframes
        f.write('%s\n' %f1_score)




if __name__ == '__main__':
    score_model()