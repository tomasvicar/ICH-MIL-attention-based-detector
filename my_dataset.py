import numpy as np
import torch


from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw



class MyDataset(torch.utils.data.Dataset):
    
    @staticmethod
    def data_tranform(data):
        
        data = data.astype(np.float32)
        data = (data-600)/600
        data = np.expand_dims(data, axis=0).copy()
        data = torch.from_numpy(data)
        
        return data
    
    
    
    @staticmethod
    def data_augmentation(data):
        
        
        
        data = data.astype(np.float32)
        data = (data-600)/600
        data = np.expand_dims(data, axis=0).copy()
        data = torch.from_numpy(data)
        
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
        
        
        
        
        
        
        
        









