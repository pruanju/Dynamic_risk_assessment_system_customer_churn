import pandas as pd
import numpy as np
import os
import json
from datetime import datetime



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    datasets = [x for x in os.listdir(os.getcwd()+'/'+input_folder_path+'/') if x[-4:]=='.csv']
    #datasets = os.listdir(os.getcwd()+'/'+input_folder_path)
    df_final = pd.DataFrame()
    for dataset in datasets: 
        df = pd.read_csv(os.getcwd()+'/'+input_folder_path+'/'+dataset) 
        df_final = df_final.append(df).reset_index(drop=True)
    
    # Delete the duplicates from the merged dataframe
    df_final.drop_duplicates(inplace=True)
    # Save the final dataframe to disk
    df_final.to_csv(os.getcwd()+'/'+output_folder_path+'/'+'finaldata.csv', index=False)

    # Save the list of csv files to disk
    with open(os.getcwd()+'/'+output_folder_path+'/'+'ingestedfiles.txt', 'w+') as f:
        # Write elements of the list of dataframes
        for items in datasets:
            f.write('%s\n' %items)
     
     



if __name__ == '__main__':
    merge_multiple_dataframe()
