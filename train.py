from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import pandas as pd


from config import Config
from dataset import Dataset
from utils.log import Log
from small_resnet3D import Small_resnet3D
from utils.losses import wce
from utils.log import get_lr



if __name__ == '__main__':

    device = torch.device("cuda:0")
    
    
    if not os.path.exists(Config.tmp_save_dir):
        os.mkdir(Config.tmp_save_dir)
    
    
    
    df = pd.read_csv(Config.data_table_path,delimiter=';')
    
    
    file_names = df['Name'].tolist()
    Kernel = df['Kernel'].to_numpy()
    Hemorrhage = df['Hemorrhage'].to_numpy()
    Fracture = df['Fracture'].to_numpy()
    
    use = Kernel == 0
    file_names = [file_names[i] for i in range(len(use)) if use[i]]
    Kernel = Kernel[use]
    Hemorrhage = Hemorrhage[use]
    Fracture = Fracture[use]  
    
    file_names = [Config.data_path + os.sep + file_name + '.mhd' for file_name in file_names]
        
    labels = np.stack((Hemorrhage,Fracture),axis=1)
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
    
    
    lbl_counts = np.sum(labels_train,axis=0)
    num_files = len(file_names_train)
    
    w_positive=num_files/lbl_counts
    w_negative=num_files/(num_files-lbl_counts)
    
    w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)
    
    
    
    loader = Dataset(split='train',file_names=file_names_train,labels=labels_train,crop_size=Config.crop_size_train)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)
    
    loader =  Dataset(split='valid',file_names=file_names_valid,labels=labels_valid,crop_size=Config.crop_size_valid)
    validLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=False,drop_last=False)
    
    
    
    model = Small_resnet3D(input_size=1, output_size=len(w_positive)).to(device)
    
     
    optimizer = optim.Adam(model.parameters(),lr=Config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)
    
    log = Log(names=['loss','acc'])
    
    for epoch_num in range(Config.max_epochs):
        
        model.train()
        for it, (batch,lbls) in enumerate(trainloader):
            
            
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res,heatmap = model(batch)
            
            res = torch.sigmoid(res)
            loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc = np.mean(((res>0.5)==(lbls>0.5)).astype(np.float32))
            
            log.append_train([loss,acc])
            
            
            
        model.eval()   
        for it, (batch,lbls) in enumerate(validLoader): 
            
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res,heatmap = model(batch)
            
            res = torch.sigmoid(res)
            loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
            
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc = np.mean(((res>0.5)==(lbls>0.5)).astype(np.float32))
            
    
            log.append_valid([loss,acc])
        
        log.save_and_reset()
    
    
        info= str(epoch_num) + '_' + str(get_lr(optimizer)) + '_train_'  + str(log.train_logs['acc'][-1]) + '_valid_' + str(log.valid_logs['acc'][-1]) 
    
        print(info)
        
        
        scheduler.step()



        tmp_file_name= Config.tmp_save_dir + os.sep +Config.model_name + info
        torch.save(model.state_dict(),tmp_file_name +  '_model.pt')
        
        
        log.plot(save_name = tmp_file_name +  '_plot.png')
        
        
        
        with open(tmp_file_name +  '_log.pkl', 'wb') as f:
            pickle.dump(log, f)
            
        with open(tmp_file_name +  '_config.pkl', 'wb') as f:
            pickle.dump(Config(), f)
    
    
    
    
    
