import numpy as np
import pandas as pd
import os

from config import Config

def read_filenames_and_labels():


    df = pd.read_csv(Config.data_table_path,delimiter=';')
    
    
    file_names = df['name'].tolist()
    # Hemorrhage = df['ICH'].to_numpy()
    # Fracture = df['Fracture'].to_numpy()
    labels = df['ICH'].to_numpy().reshape(-1,1)
    
    
    file_names = [Config.data_path + os.sep + file_name + '.mhd' for file_name in file_names]
        
    # labels = np.stack((Hemorrhage,Fracture),axis=1)
    labels = np.split(labels,labels.shape[0],axis=0)
    labels = [label[0,:] for label in labels]
    
    
    num_files =len(file_names)
    state=np.random.get_state()
    np.random.seed(42)
    split_ratio_ind = int(np.floor(Config.SPLIT_RATIO[0] / (Config.SPLIT_RATIO[0] + Config.SPLIT_RATIO[1]) * num_files))
    permuted_idx = np.random.permutation(num_files)
    train_ind = permuted_idx[:split_ratio_ind]
    valid_ind = permuted_idx[split_ratio_ind:]
    
    
    file_names_train = [file_names[i] for i in range(len(file_names)) if i in train_ind]
    labels_train = [labels[i] for i in range(len(file_names)) if i in train_ind]
    
    file_names_valid = [file_names[i] for i in range(len(file_names)) if i in valid_ind]
    labels_valid = [labels[i] for i in range(len(file_names)) if i in valid_ind]
    
    
    return file_names_train,labels_train,file_names_valid,labels_valid