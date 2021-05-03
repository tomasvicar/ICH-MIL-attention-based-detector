import numpy as np
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


class Wrapper(object):
    def __init__(self, min_h, min_d, thr):
        self.min_h = min_h
        self.min_d = min_d
        self.thr = thr
    def  __call__(self, hm_list, bb_list):
        return get_conting_value(hm_list, bb_list, self.min_h, self.min_d, self.thr)
 
    
def get_conting_value(hm_name, bb_name, min_h, min_d, thr):
        
    # save_valid_hm = r"D:\Users\Nemcek\validationRes\validationHeatmaps"
    # # save_valid_im = r"D:\Users\Nemcek\validationRes\validationImages"
    # save_valid_info = r"D:\Users\Nemcek\validationRes\validationInfo"
    
    # save_valid_hm = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\valRes_a\validationAttention"
    # save_valid_info = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\valRes_a\validationInfo"
    
    save_valid_hm = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\testRes_a\testAttention"
    save_valid_info = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\testRes_a\testInfo"
    
    
    TP = 0
    FP = 0
    FN = 0
    
    min_d = int(min_d)
    
    hm = np.load(save_valid_hm + os.sep + hm_name)
    bb = np.load(save_valid_info + os.sep + bb_name)
    bb = np.round(bb/2)
    bb_sz= bb.shape

    MyDetector = Detector(hm, min_h, min_d, thr)
    hemo_coords = MyDetector.detect()
    
    DiceObj = Dice_metrik(hm,hemo_coords,bb)
    tp,fp,fn = DiceObj.contingencyTab_bboxIndividual()
       

    out = [tp, fp, fn]
    return out

def get_dice_val(min_h, min_d, thr):
    # save_valid_hm = r"D:\Users\Nemcek\validationRes\validationHeatmaps"
    # # save_valid_im = r"D:\nemcek\EMBC2021\validationRes\validationImages"
    # save_valid_info = r"D:\Users\Nemcek\validationRes\validationInfo"

    # save_valid_hm = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\valRes_a\validationAttention"
    # save_valid_info = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\valRes_a\validationInfo"

    save_valid_hm = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\testRes_a\testAttention"
    save_valid_info = r"\\deza.feec.vutbr.cz\grp\UBMI\VYUKA\UCITEL\MLR\MOZOG\CQ_materials\testRes_a\testInfo"
    

    hms = os.listdir(save_valid_hm)
    # ims = os.listdir(save_valid_im)
    bbs = os.listdir(save_valid_info)
    
    with Pool() as pool:
        out = pool.starmap(Wrapper(min_h,min_d,thr),zip(hms,bbs))
        # print(out)
    
    out = np.array(out)
    tp_fp_fn = np.sum(out, axis=0) 
    # print(tp_fp_fn)
    
    TP = tp_fp_fn[0]
    FP = tp_fp_fn[1]
    FN = tp_fp_fn[2]
    
    dice = 2*TP / (2*TP + FP + FN)
    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    return dice, Se, PPV
        
if __name__ == '__main__':
    plt.close('all')
      
    
    min_h = 0.00378
    min_d = 58
    thr = 0.02417
    
    
    dice,Se,PPV = get_dice_val(min_h, min_d, thr)
    print(dice)
    print(Se)
    print(PPV)
    
    
    # def func(all_results=False,**params):
    #     ## pomocná funkce aby se stím líp pracovalo
    #     value = get_dice_val(params['min_h'],params['min_d'],params['thr'])
    #     return value
    
    # param_names=['min_h','min_d','thr']
    
    # bounds_lw=[0,1,0]
    # bounds_up=[0.06,60,0.03]
    
    # pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up))) 
    
    # optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
    # optimizer.maximize(init_points=10,n_iter=150)
    
    # print(optimizer.max)
    # params=optimizer.max['params']
    # print(params)