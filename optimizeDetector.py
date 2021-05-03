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


from hemorrhage_detector import Detector
from hemorrhage_detector import Dice_metrik


plt.close('all')


# min_h = 1.23428
# min_h = 0.0093
# min_d = 51
# thr = 1

def get_dice_value(min_h, min_d, thr):
    save_valid_hm = r"D:\nemcek\EMBC2021\testRes\testHeatmaps"
    save_valid_im = r"D:\nemcek\EMBC2021\testRes\testImages"
    save_valid_info = r"D:\nemcek\EMBC2021\testRes\testInfo"
    
    hms = os.listdir(save_valid_hm)
    ims = os.listdir(save_valid_im)
    bbs = os.listdir(save_valid_info)        

    TP = 0
    FP = 0
    FN = 0
    
    min_d = int(min_d)
    
    for i in range(len(ims)):
        im = np.load(save_valid_im + os.sep + ims[i])
        hm = np.load(save_valid_hm + os.sep + hms[i])
        bb = np.load(save_valid_info + os.sep + bbs[i])
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
    
    
    DICE = DiceObj.dice_metrik(TP, FP, FN)
    DICE = DICE
    return DICE

def func(all_results=False,**params):
    ## pomocná funkce aby se stím líp pracovalo
    value = get_dice_value(params['min_h'],params['min_d'],params['thr'])
    return value


# dice = get_dice_value(min_h, min_d, thr)
# print(dice)

param_names=['min_h','min_d','thr']

bounds_lw=[0,1,0]
bounds_up=[1.3,80,2]

pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up))) 

optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
optimizer.maximize(init_points=10,n_iter=120)

print(optimizer.max)
params=optimizer.max['params']
print(params)