import numpy as np
import pandas as pd
import os
import json


from config import Config

def read_filenames_and_BBs():


    df = pd.read_csv(Config.data_table_path,delimiter=';')
    
    
    file_names = df['Path'].tolist()
    BBs = df['BBs'].to_list()
    Pat_ind = df['Pat_ind'].to_numpy()
    
    file_names = [Config.data_path + os.sep + file_name for file_name in file_names]
    BoundingBoxes = []
    for i in range(len(BBs)):  
        patientBBs = []
        if isinstance(BBs[i], str):            
            bbList = BBs[i].split(sep = ';')
            for ii in range(len(bbList)-1):
                bbJason = json.loads(bbList[ii].replace('\'', '\"' ))
                patientBBs.append(np.array([bbJason['x'],bbJason['y'],bbJason['width'],bbJason['height']]))          
        BoundingBoxes.append(np.array(patientBBs))
    
    labels = [np.array(isinstance(label, str)).astype(np.float32).reshape(-1)  for label in BBs]
    
    u = np.unique(Pat_ind)
    
    
    num_files =len(u)
    np.random.seed(42)
    split_ratio_ind = int(np.floor(Config.split_valid_test[0] / (Config.split_valid_test[0] + Config.split_valid_test[1]) * num_files))
    permuted_idx = np.random.permutation(num_files)
    valid_ind_pat = u[permuted_idx[:split_ratio_ind]]
    test_ind_pat = u[permuted_idx[split_ratio_ind:]]
    
    
    valid_ind = np.argwhere(np.isin(Pat_ind,valid_ind_pat))
    test_ind = np.argwhere(np.isin(Pat_ind,test_ind_pat))
    
    
    
    
    file_names_valid = [file_names[i] for i in range(len(file_names)) if i in valid_ind]
    labels_valid = [labels[i] for i in range(len(file_names)) if i in valid_ind]
    BBs_valid = [BoundingBoxes[i] for i in range(len(file_names)) if i in valid_ind]
    
    
    file_names_test = [file_names[i] for i in range(len(file_names)) if i in test_ind]
    labels_test = [labels[i] for i in range(len(file_names)) if i in test_ind]
    BBs_test = [BoundingBoxes[i] for i in range(len(file_names)) if i in test_ind]

    
    # file_names = [file_names[i] for i in range(len(file_names))]
    # labels = [labels[i] for i in range(len(file_names))]
    
   
    
    
    return file_names_valid, labels_valid, BBs_valid, file_names_test, labels_test, BBs_test  
