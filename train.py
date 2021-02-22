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
import json
import matplotlib.pyplot as plt
from skimage.transform import resize



from config import Config
from my_dataset import MyDataset
from utils.log import Log
from utils.losses import wce
from utils.log import get_lr
from read_filenames_and_labels import read_filenames_and_labels



if __name__ == '__main__':

    device = torch.device("cuda:0")
    
    
    if not os.path.exists(Config.tmp_save_dir):
        os.mkdir(Config.tmp_save_dir)
    
    
    
    file_names_train,labels_train,file_names_valid,labels_valid = read_filenames_and_labels()

    lbl_counts = np.sum(labels_train,axis=0)
    num_files = len(file_names_train)
    
    w_positive=num_files/lbl_counts
    w_negative=num_files/(num_files-lbl_counts)
    
    w_positive = np.array(w_positive).reshape(-1)
    w_negative = np.array(w_negative).reshape(-1)

    w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)
    
    
    loader = MyDataset(split='train',file_names=file_names_train,labels=labels_train)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)
    
    loader =  MyDataset(split='valid',file_names=file_names_valid,labels=labels_valid)
    validLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=True,drop_last=True)
    
    
    model = Config.net(input_size=3, output_size=len(w_positive)).to(device)
    
     
    optimizer = optim.Adam(model.parameters(),lr=Config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=Config.lr_steps, gamma=Config.gamma, last_epoch=-1)
    
    log = Log(names=['loss','acc'])
    
    for epoch_num in range(Config.max_epochs):
        
        
        N = len(trainloader)
        for it, (batch,lbls) in enumerate(trainloader):
                   
            model.train()
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
            
            
            if (it % int(N/Config.plots_in_epoch) == 0) and (it != 0):
                model.eval()   
                with torch.no_grad():
                    for itt, (batch,lbls) in enumerate(validLoader): 

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
            
                info= str(epoch_num) + '_' + str(it) + '_' + str(get_lr(optimizer)) + '_train_'  + str(log.train_logs['acc'][-1]) + '_valid_' + str(log.valid_logs['acc'][-1]) 
            
                print(info)
                
                tmp_file_name= Config.tmp_save_dir + os.sep +Config.model_name + info
                log.plot(save_name = tmp_file_name +  '_plot.png')
                
                
                batch = batch.detach().cpu().numpy()
                heatmap = heatmap.detach().cpu().numpy()
                
                for k in range(batch.shape[0]):
                    res_tmp = res[k,0]
                    lbl_tmp = lbls[k,0]
                    img_tmp = batch[k,1,:,:]
                    heatmap_tmp = heatmap[k,0,:,:]
                    heatmap_tmp = resize(heatmap_tmp,img_tmp.shape)
                    
                    plt.figure(figsize=[6.4*3, 4.8*3])
                    plt.subplot(121)
                    plt.imshow(img_tmp)
                    plt.title(str(k) + '  gt=' + str(lbl_tmp) + '  res=' + str(res_tmp))
                    
                    plt.subplot(122)
                    plt.imshow(heatmap_tmp)
                    plt.savefig(Config.tmp_save_dir + os.sep +Config.model_name + info + '_example_image' + str(k) + '.png')
                    plt.show()
                    plt.close()
        
        
        
        
        
        
        
        

        
        
        
        info= str(epoch_num) + '_' + str(get_lr(optimizer)) + '_train_'  + str(log.train_logs['acc'][-1]) + '_valid_' + str(log.valid_logs['acc'][-1]) 
        
        print(info)
        
        scheduler.step()



        tmp_file_name= Config.tmp_save_dir + os.sep +Config.model_name + info
        torch.save(model.state_dict(),tmp_file_name +  '_model.pt')
        

        with open(tmp_file_name +  '_log.pkl', 'wb') as f:
            pickle.dump(log, f)
            
        with open(tmp_file_name +  '_config.pkl', 'wb') as f:
            pickle.dump(Config(), f)
    
        with open(tmp_file_name +  'filenames_and_lbls.json', 'w') as f:
            filenames_and_lbls = {'file_names_train':file_names_train,'labels_train':np.stack(labels_train,axis=0).tolist(),
                                  'file_names_valid':file_names_valid,'labels_valid':np.stack(labels_valid,axis=0).tolist()}
            json.dump(filenames_and_lbls, f, indent=2) 
    
    
    
    
