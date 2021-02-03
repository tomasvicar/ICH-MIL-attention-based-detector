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


from config import Config
from dataset import Dataset
from utils.log import Log
from small_resnet3D import Small_resnet3D
from utils.losses import wce
from log import get_lr



if __name__ == '__main__':

    device = torch.device("cuda:0")
    
    
    if not os.path.exists(Config.tmp_save_dir):
        os.mkdir(Config.tmp_save_dir)
    
    
    file_names_train = []
    labels_train = []
    
    file_names_valid = []
    labels_valid = []
    
    w_positive_tensor = []
    w_negative_tensor = []
    
    
    
    loader = Dataset(split='train',file_names=file_names_train,labels=labels_train,crop_size=Config.crop_size_train)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)
    
    loader =  Dataset(split='valid',file_names=file_names_valid,labels=labels_valid,crop_size=Config.crop_size_valid)
    validLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=False,drop_last=False)
    
    
    
    model = Small_resnet3D().to(device)
    
    
    optimizer = optim.Adam(model.parameters(),lr=Config. init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)
    
    log = Log()
    
    for epoch_num in range(Config.max_epochs):
        
        model.train()
        for it, (batch,lbls) in enumerate(trainloader):
            
            
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            
            res=torch.sigmoid(res)
            loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc=np.mean((np.argmax(res,1)==np.argmax(lbls,1)).astype(np.float32))
            
            log.append_train(loss,acc)
            
            
            
        model.eval()   
        for it, (batch,lbls) in enumerate(validLoader): 
            
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            
            res=torch.sigmoid(res)
            loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
            
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc=np.mean((np.argmax(res,1)==np.argmax(lbls,1)).astype(np.float32))
            
    
            log.append_test(loss,acc)
        
        log.save_and_reset()
    
    
        info= str(epoch_num) + '_' + str(get_lr(optimizer)) + '_train_'  + str(log.trainig_acc_log[-1]) + '_valid_' + str(log.valid_acc_log[-1]) 
    
        print(info)
        log.plot()
        
        
        scheduler.step()

        tmp_file_name= Config.tmp_save_dir + os.sep +Config.model_name + info
        torch.save(model.state_dict(),tmp_file_name +  '_model.pt')
        log.save_plot(tmp_file_name +  '_plot.png')
        
        with open(tmp_file_name +  '_log.pkl', 'wb') as f:
            pickle.dump(log, f)
            
        with open(tmp_file_name +  '_config.pkl', 'wb') as f:
            pickle.dump(Config(), f)
    
    
    
    
    
