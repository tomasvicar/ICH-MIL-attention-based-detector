import numpy as np
import torch
import SimpleITK as sitk
import random
from datetime import datetime

from config import Config
from utils.load_dicom_slice import load_dicom_slice
from utils.get_dicom_slice_size import get_dicom_slice_size
from utils.matrixDeformer import MatrixDeformer
from utils.blurSharpAugmenter import BlurSharpAugmenter




class MyDataset(torch.utils.data.Dataset):
    
    @staticmethod
    def data_tranform(data):
        
        data = data
        data[data<-500] = 0
        data = data.astype(np.float32).copy()
        
        
        ##### Subdural window
        data_sd = data.copy()
        data_sd = ((data_sd-1009.0)/(1139.0-1009.0)) * (2**12)
        data_sd[data_sd<0.0] = 0.0
        data_sd[data_sd>2**12] = 2**12
        ##### Brain window
        data_br = data.copy()
        data_br = ((data_br-1024.0)/(1104.0-1024.0)) * (2**12)
        data_br[data_br<0.0] = 0.0
        data_br[data_br>2**12] = 2**12
        

        
        data = np.concatenate((np.expand_dims(data, axis=0),
                                np.expand_dims(data_br, axis=0),
                                np.expand_dims(data_sd, axis=0)), axis=0)

        data = (data-400)/1000

        data = torch.from_numpy(data)
        
        return data
    
    
    
    @staticmethod
    def data_augmentation(data):
        
        if random.random() < Config.global_p:
            ## multiply augmentation
            if random.random() < Config.individual_p:
                max_multiplier  = Config.max_multiplier        
                multiplier = 1 + random.random() * max_multiplier
                if random.random()>0.5:
                    multiplier = 1 / multiplier
                    
                data = data * multiplier
            
            
            ## add augmentation
            if random.random() < Config.individual_p:
                max_add = Config.max_add
                add_value = max_add - 2 * max_add * random.random() 
                data = data + add_value
            
            
            ## afine transformation
            if random.random() < Config.individual_p:
                md = MatrixDeformer(scale_range=Config.scale_range,shears_range=Config.shears_range,tilt_range=Config.tilt_range,
                                    translation_range=Config.translation_range,rotation_range=Config.rotation_range)
                data = md.augment(data)
            
            if random.random() < Config.individual_p:
                bs = BlurSharpAugmenter(Config.blur_sharp_range)
                data = bs.augment(data)
        
        
        if random.random() > 0.5:
            data = data[:,::-1]
        
        
        return data
     
        
    @staticmethod
    def label_tranform(label):
        
        label = label.astype(np.float32)
        label = torch.from_numpy(label)
        
        return label
     
        
    
    
    def __init__(self, split,file_names,labels):
        
        
        self.split=split
        self.labels=labels
        self.file_names = file_names

            
            
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        
        file_name = self.file_names[index]
        
        lbl = self.labels[index]
        

        try:
            img = load_dicom_slice(file_name)
        except:
            img = np.zeros((256,256))
            with open('error_read_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f") + '.txt', 'w') as f:
                f.write(file_name)
        
        
        if (img.shape[0]!=256) or (img.shape[1]!=256):
            img = np.zeros((256,256))
            with open('error_size_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f") + '.txt', 'w') as f:
                f.write(file_name)
            
        
        img = img.astype(np.float32)
        
        if self.split == 'train':
            img = self.data_augmentation(img)
        
        
        
        img = self.data_tranform(img)
        
        lbl = self.label_tranform(lbl)
          
        return img, lbl
        
        
        
        
        
        
        
        









