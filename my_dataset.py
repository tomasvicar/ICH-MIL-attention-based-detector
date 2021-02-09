import numpy as np
import torch
import napari
import SimpleITK as sitk
import random
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from config import Config


from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw



class MyDataset(torch.utils.data.Dataset):
    
    @staticmethod
    def data_tranform(data):
        
        data = data.astype(np.float32).copy()
        
        data_sd = data.astype(np.float32).copy()
        data_sd = ((data-1009.0)/(1139.0-1009.0)) * (2^12)
        data_sd[data_sd<0.0] = 0.0
        data_sd[data_sd>2^12] = 2^12
        data_sd = data_sd.astype(np.uint16)
        
        data_br = data.astype(np.float32).copy()
        data_br = ((data-1024.0)/(1104.0-1024.0)) * (2^12)
        data_br[data_sd<0.0] = 0.0
        data_br[data_sd>2^12] = 2^12
        data_br = data_sd.astype(np.uint16)
        
        # data = (data-600)/600 transformuj do 0-1
        # np.concatenate
        # data = np.expand_dims(data, axis=0).copy() # pridanie prazdnej dim
        data = torch.from_numpy(data)
        
        return data
    
    
    
    @staticmethod
    def data_augmentation(data):
        
        ## multiply augmentation
        max_multiplier  = Config.max_multiplier        
        multiplier = 1 + random.random() * max_multiplier
        if random.random()>0.5:
            multiplier = 1 / multiplier
            
        data = data * multiplier
        
        
        ## add augmentation
        max_add = Config.max_add
        add_value = max_add -2*random.random() 
        data = data + add_value
        
        ### Random crop
        dat_size = data.shape
        max_perc = Config.max_crop_perc
        x_max_crop = round(dat_size[0]*max_perc)
        y_max_crop = round(dat_size[1]*max_perc)
        z_max_crop = round(dat_size[2]*max_perc)
        
        Xc = random.randint(0,x_max_crop)
        Yc = random.randint(0,y_max_crop)
        Zc = random.randint(0,z_max_crop)
        
        data = data[Xc : int(dat_size[0]-x_max_crop+Xc),
                    Yc : int(dat_size[1]-y_max_crop+Yc),
                    Zc : int(dat_size[2]-z_max_crop+Zc)]
        
        #### Random resize
        max_resize_perc = Config.max_resize_perc
        x_res = random.uniform(1-max_resize_perc,1+max_resize_perc)
        y_res = random.uniform(1-max_resize_perc,1+max_resize_perc)
        z_res = random.uniform(1-max_resize_perc,1+max_resize_perc)  
        
        data = zoom(data, (x_res,y_res,z_res))
                
        #all flips and rotations
        if random.random()>0.5:
            data = data[::-1,:,:]         
        if random.random()>0.5:
            data = data[:,::-1,:]       
        if random.random()>0.5:
            data = data[:,:,::-1]    
        
        data = np.rot90(data,random.randrange(4),axes=(0,1))
        
        if random.random()>0.5:
            data = data[::-1,:,:]         
        if random.random()>0.5:
            data = data[:,::-1,:]        
        if random.random()>0.5:
            data = data[:,:,::-1]

        # Axial rotation
        max_rot_angle = Config.max_rot_angle
        data = rotate(data, angle=random.randrange(max_rot_angle), axes = (0,1)) 

        return data
     
        
    @staticmethod
    def label_tranform(label):
        
        label = label.astype(np.float32)
        label = torch.from_numpy(label)
        
        return label
     
        
    
    
    def __init__(self, split,file_names,labels,crop_size=[480,480,25]):
        
        
        self.split=split
        self.labels=labels
        self.crop_size=crop_size
        self.file_names = file_names
        
        self.sizes = [get_size_raw(file_name) for file_name in self.file_names]
        

            
            
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        
        crop_size = self.crop_size
        file_name = self.file_names[index]
        size = self.sizes[index]
        
        lbl = self.labels[index]
        
        
        
        if not self.crop_size==None:
            p = -1*np.ones(3)
            p[0]=torch.randint(size[0]-crop_size[0],(1,1)).view(-1).numpy()[0]
            p[1]=torch.randint(size[1]-crop_size[1],(1,1)).view(-1).numpy()[0]
            p[2]=torch.randint(size[2]-crop_size[2],(1,1)).view(-1).numpy()[0]
            
            img = read_raw(file_name,crop_size,[int(p[0]),int(p[1]),int(p[2])])
        else:
            img = read_raw(file_name)
        
        
        img = img.astype(np.flot32 )
        
        
        
        img = self.data_augmentation(img)
        
        
        img = self.data_tranform(img)
        
        lbl = self.label_tranform(lbl)
          
        return img, lbl
        
        
        
        
        
        
        
        









