import torch
import numpy as np   


def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)