import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df = pd.read_csv(os.getcwd()+'/'+test_data_path+'/'+'testdata.csv')
    y_pred = model_predictions(df)
    y_real = df['exited'].to_numpy()

    cm = metrics.confusion_matrix(y_real, y_pred)

    # Confusion matrix plot inspired by https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

    names = ["TN","FP","FN","TP"]
    numbers = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in  zip(names,numbers)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')
    plt.savefig(os.getcwd()+'/'+model_path+'/'+'confusionmatrix.png')





if __name__ == '__main__':
    score_model()
