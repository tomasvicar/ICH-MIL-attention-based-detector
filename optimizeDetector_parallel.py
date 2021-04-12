import numpy as np
import torch
from torch.utils import data
import os
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pickle
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.measure import label
from skimage import color
from bayes_opt import BayesianOptimization
from multiprocessing import Pool



from hemorrhage_detector import Detector
from hemorrhage_detector import Dice_metrik


plt.close('all')



class Wrapper(object):
    def __init__(self, min_h, min_d, thr):
        self.min_h = min_h
        self.min_d = min_d
        self.thr = thr
    def  __call__(self, hm_name, bb_name):
        return get_conting_value(hm_name, bb_name, self.min_h, self.min_d, self.thr)
 

def get_conting_value(hm_name, bb_name, min_h, min_d, thr):
    
    save_valid_hm = r"D:\nemcek\EMBC2021\validationRes\validationHeatmaps"
    save_valid_im = r"D:\nemcek\EMBC2021\validationRes\validationImages"
    save_valid_info = r"D:\nemcek\EMBC2021\validationRes\validationInfo"
    
    TP = 0
    FP = 0
    FN = 0
    
    min_d = int(min_d)
    
    # im = np.load(save_valid_im + os.sep + ims[i])
    hm = np.load(save_valid_hm + os.sep + hm_name)
    bb = np.load(save_valid_info + os.sep + bb_name)
    bb = np.round(bb/2)
    bb_sz= bb.shape
  
    # if bb_sz[1] > 0:
    #     plt.figure()
    #     plt.imshow(im)
    #     ax = plt.gca()
                
    #     for ii in range(bb_sz[1]):
    #         rect = patches.Rectangle((bb[0,ii,0],bb[0,ii,1]),
    #                                   bb[0,ii,2],bb[0,ii,3],
    #                                   linewidth=1,edgecolor='r',facecolor='none')
            
           
    #         ax.add_patch(rect)    
    #     MyDetector = Detector(hm, min_h, min_d, thr)
    #     hemo_coords = MyDetector.detect()    
    #     plt.plot(hemo_coords[:,1],hemo_coords[:,0],'*r')
            
    #     plt.show()
        


    MyDetector = Detector(hm, min_h, min_d, thr)
    hemo_coords = MyDetector.detect()
    
    DiceObj = Dice_metrik(hm,hemo_coords,bb)
    tp,fp,fn = DiceObj.contingencyTab()
    TP = TP + tp
    FP = FP + fp
    FN = FN + fn
    
    
    # plt.figure()
    # plt.imshow(im, cmap='gray')
    # plt.plot(hemo_coords[:,1],hemo_coords[:,0],'*r')
    # plt.show()
    
    
    # DICE = DiceObj.dice_metrik(TP, FP, FN)
    # DICE = DICE
    out = [TP, FP, FN]
    return out

def get_dice_val(min_h, min_d, thr):
    save_valid_hm = r"D:\nemcek\EMBC2021\validationRes\validationHeatmaps"
    save_valid_im = r"D:\nemcek\EMBC2021\validationRes\validationImages"
    save_valid_info = r"D:\nemcek\EMBC2021\validationRes\validationInfo"

    hms = os.listdir(save_valid_hm)
    ims = os.listdir(save_valid_im)
    bbs = os.listdir(save_valid_info) 
    
    with Pool() as pool:
        out = pool.map(Wrapper(min_h,min_d,thr),hms,bbs)
        print(out)



min_h = 0.0000001
min_d = 20
thr = 0.2
get_dice_val(min_h, min_d, thr)


# def func(all_results=False,**params):
#     ## pomocná funkce aby se stím líp pracovalo
#     value = get_dice_value(params['min_h'],params['min_d'],params['thr'])
#     return value

# param_names=['min_h','min_d','thr']

# bounds_lw=[0,1,-1]
# bounds_up=[2,100,4]

# pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up))) 

# optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
# optimizer.maximize(init_points=10,n_iter=50)

# print(optimizer.max)
# params=optimizer.max['params']
# print(params)