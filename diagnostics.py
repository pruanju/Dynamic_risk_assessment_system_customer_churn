
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions(dataframe):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.getcwd()+'/'+model_path+'/'+'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    X = dataframe.drop(columns=['corporation', 'exited'], axis=1).to_numpy()

    y_pred = model.predict(X)
    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    df = pd.read_csv(os.getcwd()+'/'+dataset_csv_path+'/'+'finaldata.csv')
    df_final = df.drop(columns=['corporation', 'exited'], axis=1)
    mean = df_final.mean(axis=0)
    median = df_final.median(axis=0)
    std = df_final.std(axis=0)
    summary = []
    for col in df_final.columns.values:
        mean_ = mean[col]
        median_ = median[col]
        std_ = std[col]
        line = [col, mean_, median_, std_]
        summary.append(line) 

    return summary

##################Function to count number of nulls
def missing_data():
    """
    Statistics on null data
    """
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    df = pd.read_csv(os.getcwd()+'/'+dataset_csv_path+'/'+'finaldata.csv')

    num_nulls = df.isna().sum()
    results = [num_nulls[i]/len(df.index) for i in range(len(num_nulls))]
    column_names = df.columns.values
    
    summary = []
    for x,y in zip(column_names,results):
        line = [x, y]
        summary.append(line) 

    return summary

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    summary = []

    starttime = timeit.default_timer() 
    os.system('python training.py') 
    timing=timeit.default_timer() - starttime 
    summary.append(['training', timing])

    starttime = timeit.default_timer() 
    os.system('python ingestion.py') 
    timing=timeit.default_timer() - starttime 
    summary.append(['ingestion', timing])

    return summary

##################Function to check dependencies
def outdated_packages_list():
    """
    Use requirements.txt to list current and latest version of required packages
    """


    pip_outdated_binary = subprocess.Popen(['pip', 'list','--outdated'], stdout=subprocess.PIPE)
    pip_outdated = []
    while True:
        output = pip_outdated_binary.stdout.readline()
        if output.decode('utf8') == '' and pip_outdated_binary.poll() is not None:
            break
        if output:
            pip_outdated.append(output.decode('utf8').strip().split())
    rc = pip_outdated_binary.poll()

    

    # read current requirements file
    with open("requirements.txt", 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]

    summary = []
    for l in lines:
        x = l.split("==")
        for y in pip_outdated:
            if x[0] == y[0]:
                summary.append(f"{y[0]} - {y[1]} - {y[2]}")
    print(summary)
    return summary


if __name__ == '__main__':
    df = pd.read_csv(os.getcwd()+'/'+test_data_path+'/'+'testdata.csv')
    predictions = model_predictions(df)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
