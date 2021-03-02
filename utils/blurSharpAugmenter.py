import random
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import laplace


class BlurSharpAugmenter():
    def __init__(self,bs_r=(-0.5,0.5)): 
        
        r=1-2*rand()

        
        if r<=0:
            self.type='s'
            self.par=bs_r[0]*r
            
        if r>0:
            self.type='b'
            self.par=bs_r[1]*r
        
        
        
    def augment(self,img):   
        if self.type=='b':
            img = gaussian_filter(img,self.par)
        
        if self.type=='s':
            img = img-self.par*laplace(img)
        
    
        return img

def rand():
    return random.random()