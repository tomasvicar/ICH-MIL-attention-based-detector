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
from my_dataset import MyDataset
from utils.log import Log
from utils.losses import wce
from utils.log import get_lr
from my_dataset import MyDataset
from utils.load_dicom_slice import load_dicom_slice
from resnet_2D_heatmap import Resnet_2D_heatmap
from read_filenames_and_labels import read_filenames_and_labels




device = torch.device("cuda:0")

model_name = r"D:\nemcek\EMBC2021\rsna_network\model29_1e-05_train_0.9085252_valid_0.9016779"

with open(model_name + 'filenames_and_lbls.json', 'r') as f:
    filenames_and_lbls = json.load(f)
    
file_names,labels = read_filenames_and_labels()
lbl_counts = np.sum(labels,axis=0)
num_files = len(file_names)

w_positive=num_files/lbl_counts
w_negative=num_files/(num_files-lbl_counts)

w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)


with open(model_name +  '_config.pkl', 'rb') as f:
    config = pickle.load(f)
    
    

model = Resnet_2D_heatmap(input_size=3, output_size=1)

model.load_state_dict(torch.load(model_name + '_model.pt'))

model = model.to(device)


loader = MyDataset(split='valid',file_names=file_names,labels=labels)
validLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=True,drop_last=True)

log = Log(names=['loss','acc'])

model.eval()
with torch.no_grad():
    N = len(validLoader)
    for it, (batch,lbls) in enumerate(validLoader):      
        if (it % 50) == 0:
            print('valid  ' +  str(it) + ' / ' + str(N))
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
        
        if it > 0:
            break

log.save_and_reset()
info= 'cq_' + str(log.valid_logs['acc'][-1]) 

batch = batch.detach().cpu().numpy()
heatmap = heatmap.detach().cpu().numpy()


saveImHMpath = r"D:\nemcek\EMBC2021\test_Imgs_Heatmaps_detector"

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

    if lbl_tmp == 1 and res_tmp > 0.5:
        np.save(saveImHMpath + '\img_' + str(k), img_tmp )
        np.save(saveImHMpath + '\hm_' + str(k), heatmap_tmp )