import napari
import numpy as np
import torch
from torch.utils import data
import os
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import json
import pickle


from my_dataset import MyDataset
from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw
from small_resnet3D import Small_resnet3D




index = 7

model_name = r"D:\vicar\kuba_embc2021\models_python\model36_0.001_train_0.34313726_valid_0.32608697"


with open(model_name + 'filenames_and_lbls.json', 'r') as f:
    filenames_and_lbls = json.load(f)

file_names_valid =  filenames_and_lbls['file_names_valid']
labels_valid =  filenames_and_lbls['labels_valid']
labels_valid = [np.array(x) for x in labels_valid]

with open(model_name +  '_config.pkl', 'rb') as f:
    config = pickle.load(f)




file_name = file_names_valid[index]
labels_valid = labels_valid[index]


device = torch.device("cuda:0")


model = Small_resnet3D(input_size=1, output_size=2)


model.load_state_dict(torch.load(model_name + '_model.pt'))

model = model.to(device)
model.eval()



crop_size = config.crop_size_valid
size = get_size_raw(file_name)
        
        




img_whole = read_raw(file_name)
w_whole = np.zeros_like(img_whole,dtype=np.float32)
heatmap_hemo_whole = np.zeros_like(img_whole,dtype=np.float32)
heatmap_frac_whole = np.zeros_like(img_whole,dtype=np.float32)
img_whole_check = np.zeros_like(img_whole,dtype=np.float32)

overlap = 5;
border = overlap
zstep = 25 
z_size = img_whole.shape[2]

w = np.ones((img_whole.shape[0],img_whole.shape[1],zstep),dtype=np.float32)
tmp = np.ones((overlap,overlap,overlap))
tmp = tmp/np.sum(tmp)
w = convolve(w,tmp,mode='constant')



start_inds = [0]
actual_ind = 0
stop = 0
while not stop:
    actual_ind = actual_ind + zstep - overlap
    if actual_ind >= (z_size-zstep):
        actual_ind = z_size-zstep
        stop = 1
    start_inds.append(actual_ind)
    
    
    
for start_ind in start_inds:
    
    
    
    img = img_whole[:,:,start_ind:(start_ind+zstep)]
    
    img = MyDataset.data_tranform(img)
    
    img = torch.unsqueeze(img, 0)
    
    
    
    img = img.to(device)
    
    res,heatmap = model(img)
    
    res = res.detach().cpu().numpy()
    heatmap = heatmap.detach().cpu().numpy()
    img = img.detach().cpu().numpy()
    
    
    img = img[0,0,:,:,:]
    heatmap_hemo = heatmap[0,0,:,:,:]
    heatmap_frac = heatmap[0,1,:,:,:]
                                
    
    
    heatmap_hemo = zoom(heatmap_hemo,np.array(img.shape)/np.array(heatmap_hemo.shape))
    heatmap_frac = zoom(heatmap_frac,np.array(img.shape)/np.array(heatmap_frac.shape))

    
    w_whole[:,:,start_ind:(start_ind+zstep)] = w_whole[:,:,start_ind:(start_ind+zstep)] + w
    heatmap_hemo_whole[:,:,start_ind:(start_ind+zstep)] = heatmap_hemo_whole[:,:,start_ind:(start_ind+zstep)] + w * heatmap_hemo
    heatmap_frac_whole[:,:,start_ind:(start_ind+zstep)] = heatmap_frac_whole[:,:,start_ind:(start_ind+zstep)] + w * heatmap_frac
    img_whole_check[:,:,start_ind:(start_ind+zstep)] = img_whole_check[:,:,start_ind:(start_ind+zstep)] + w * img


heatmap_hemo_whole = heatmap_hemo_whole / w_whole
heatmap_frac_whole = heatmap_frac_whole / w_whole
img_whole_check = img_whole_check / w_whole








print('GT - hemo, frac')
print(labels_valid)
print('res - hemo, frac')
print(res[0,:])


with napari.gui_qt():
    viewer = napari.Viewer(order=[2,1,0])
    viewer.add_image(img_whole_check,name='img')
    viewer.add_image(heatmap_hemo_whole, name='heatmap_hemo')
    viewer.add_image(heatmap_frac_whole, name='heatmap_frac')    


    












