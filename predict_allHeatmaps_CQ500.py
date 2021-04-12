import numpy as np
import torch
from torch.utils import data
import os
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import json
import pickle
from skimage.transform import resize


from utils.log import Log
from config import Config
from my_dataset_BBs import MyDataset
from utils.log import Log
from utils.losses import wce
from utils.log import get_lr
from my_dataset_BBs import MyDataset
from utils.load_dicom_slice import load_dicom_slice
from resnet_2D_heatmap import Resnet_2D_heatmap
from read_filenames_and_BBs import read_filenames_and_BBs




device = torch.device("cuda:0")

model_name = r"D:\nemcek\EMBC2021\rsna_network\model29_1e-05_train_0.9217379_valid_0.9087358"

with open(model_name + 'filenames_and_lbls.json', 'r') as f:
    filenames_and_lbls = json.load(f)
    
file_names_valid, labels_valid, BBs_valid, file_names_test, labels_test, BBs_test = read_filenames_and_BBs()
# lbl_counts = np.sum(labels,axis=0)
# num_files = len(file_names)

# w_positive=num_files/lbl_counts
# w_negative=num_files/(num_files-lbl_counts)

# w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
# w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)


with open(model_name +  '_config.pkl', 'rb') as f:
    config = pickle.load(f)
    
    

model = Resnet_2D_heatmap(input_size=3, output_size=1)

model.load_state_dict(torch.load(model_name + '_model.pt'))

model = model.to(device)


loader = MyDataset(split='valid',file_names=file_names_valid,labels=labels_valid, bboxes=BBs_valid)
validLoader= data.DataLoader(loader, batch_size=Config.pred_batch_size, num_workers=Config.test_num_workers, shuffle=True,drop_last=True)

loader = MyDataset(split='valid',file_names=file_names_test,labels=labels_test, bboxes=BBs_test)
testLoader= data.DataLoader(loader, batch_size=Config.pred_batch_size, num_workers=Config.test_num_workers, shuffle=True,drop_last=True)

# log = Log(names=['loss','acc'])
save_valid_hm = r"D:\nemcek\EMBC2021\validationRes\validationHeatmaps"
save_valid_im = r"D:\nemcek\EMBC2021\validationRes\validationImages"
save_valid_info = r"D:\nemcek\EMBC2021\validationRes\validationInfo"

model.eval()
image_ind = 0
with torch.no_grad():
    N = len(validLoader)
    for it, (batch,lbls,bbs) in enumerate(validLoader):      
        if (it % 50) == 0:
            print('valid  ' +  str(it) + ' / ' + str(N))
        batch=batch.to(device)
        lbls=lbls.to(device)
        # bbs = bbs.to(device)
        
        
        
        res,heatmap = model(batch)
        
        res = torch.sigmoid(res)
        # loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)   
        # loss=loss.detach().cpu().numpy()
        
        res=res.detach().cpu().numpy()
        lbls=lbls.detach().cpu().numpy()
        bbs=bbs.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        heatmap = heatmap.detach().cpu().numpy()

        acc = np.mean(((res>0.5)==(lbls>0.5)).astype(np.float32))
        for k in range(batch.shape[0]):
            res_tmp = res[k,0]
            lbl_tmp = lbls[k,0]
            bb_tmp = bbs.copy()
            img_tmp = batch[k,1,:,:]
            heatmap_tmp = heatmap[k,0,:,:]
            heatmap_tmp = resize(heatmap_tmp,img_tmp.shape)
            
            np.save(save_valid_im + '\img_' + str(image_ind), img_tmp )
            np.save(save_valid_hm + '\hm_' + str(image_ind), heatmap_tmp )
            np.save(save_valid_info + '\info_' + str(image_ind), bb_tmp )

            image_ind = image_ind + 1


save_test_hm = r"D:\nemcek\EMBC2021\testRes\testHeatmaps"
save_test_im = r"D:\nemcek\EMBC2021\testRes\testImages"
save_test_info = r"D:\nemcek\EMBC2021\testRes\testInfo"

model.eval()
image_ind = 0
with torch.no_grad():
    N = len(testLoader)
    for it, (batch,lbls,bbs) in enumerate(testLoader):      
        if (it % 50) == 0:
            print('valid  ' +  str(it) + ' / ' + str(N))
        batch=batch.to(device)
        lbls=lbls.to(device)
        # bbs = bbs.to(device)
        
        
        
        res,heatmap = model(batch)
        
        res = torch.sigmoid(res)
        # loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)   
        # loss=loss.detach().cpu().numpy()
        
        res=res.detach().cpu().numpy()
        lbls=lbls.detach().cpu().numpy()
        bbs=bbs.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        heatmap = heatmap.detach().cpu().numpy()

        acc = np.mean(((res>0.5)==(lbls>0.5)).astype(np.float32))
        for k in range(batch.shape[0]):
            res_tmp = res[k,0]
            lbl_tmp = lbls[k,0]
            bb_tmp = bbs.copy()
            img_tmp = batch[k,1,:,:]
            heatmap_tmp = heatmap[k,0,:,:]
            heatmap_tmp = resize(heatmap_tmp,img_tmp.shape)
            
            np.save(save_test_im + '\img_' + str(image_ind), img_tmp )
            np.save(save_test_hm + '\hm_' + str(image_ind), heatmap_tmp )
            np.save(save_test_info + '\info_' + str(image_ind), bb_tmp )

            image_ind = image_ind + 1



