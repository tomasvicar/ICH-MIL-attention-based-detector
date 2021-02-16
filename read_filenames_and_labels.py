import numpy as np
import pandas as pd
import os

from config import Config

def read_filenames_and_labels():


    df = pd.read_csv(Config.data_table_path,delimiter=';')
    
    
    file_names = df['Path'].tolist()
    BBs = df['BBs'].to_list()
    Pat_ind = df['Pat_ind'].to_numpy()
    
    
    file_names = [Config.data_path + os.sep + file_name.replace('\','/') for file_name in file_names]
        
    
    labels = [np.array(isinstance(label, str)).astype(np.float32).reshape(-1)  for label in BBs]
    
    
    
    u = np.unique(Pat_ind)
    
    
    num_files =len(u)
    np.random.seed(42)
    split_ratio_ind = int(np.floor(Config.SPLIT_RATIO[0] / (Config.SPLIT_RATIO[0] + Config.SPLIT_RATIO[1]) * num_files))
    permuted_idx = np.random.permutation(num_files)
    train_ind_pat = u[permuted_idx[:split_ratio_ind]]
    valid_ind_pat = u[permuted_idx[split_ratio_ind:]]
    
    
    train_ind = np.argwhere(np.isin(Pat_ind,train_ind_pat))
    valid_ind = np.argwhere(np.isin(Pat_ind,valid_ind_pat))
    
    
    
    
    file_names_train = [file_names[i] for i in range(len(file_names)) if i in train_ind]
    labels_train = [labels[i] for i in range(len(file_names)) if i in train_ind]
    
    file_names_valid = [file_names[i] for i in range(len(file_names)) if i in valid_ind]
    labels_valid = [labels[i] for i in range(len(file_names)) if i in valid_ind]
    
    
    return file_names_train,labels_train,file_names_valid,labels_valid
