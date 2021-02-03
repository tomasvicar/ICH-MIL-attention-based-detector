import numpy as np
from torch.utils import data
import torch


from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw



class Dataset(data.Dataset):
    
    @staticmethod
    def data_tranform(data):
        
        data = data.astype(np.float32)
        data = (data-600)/600
        data = np.expand_dims(data, axis=0).copy()
        data = torch.from_numpy(data)
        
        return data
     

        
    
    
    def __init__(self, split,file_names,labels,crop_size=[128,128,25]):
        
        
        self.split=split
        self.file_names=file_names
        self.labels=labels
        self.crop_size=crop_size
        
        
        self.sizes=[]
        
        for name in self.file_names:
            self.sizes.append(get_size_raw(name))
            
            
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        
        crop_size = self.crop_size
        file_name = self.file_names[index]
        size = self.file_names[index]
        
        
        
        p = -1*np.ones(3)
        p[0]=torch.randint(size[0]-crop_size[0],(1,1)).view(-1).numpy()[0]
        p[1]=torch.randint(size[1]-crop_size[1],(1,1)).view(-1).numpy()[0]
        p[2]=torch.randint(size[2]-crop_size[2],(1,1)).view(-1).numpy()[0]
        
        
        img = read_raw(file_name,crop_size,[int(p[0]),int(p[1]),int(p[2])])
        
        img = self.data_tranform(img)
          
        return img
        
        
        
        
        
        
        
        









