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



 
    
def get_conting_value(hm_name, im_name, bb_name, min_h, min_d, thr):
        
    save_valid_hm = r"D:\Users\Nemcek\testRes\testHeatmaps"
    save_valid_im = r"D:\Users\Nemcek\testRes\testImages"
    save_valid_info = r"D:\Users\Nemcek\testRes\testInfo"
    
    TP = 0
    FP = 0
    FN = 0
    
    min_d = int(min_d)
    
    im = np.load(save_valid_im + os.sep + im_name)
    hm = np.load(save_valid_hm + os.sep + hm_name)
    bb = np.load(save_valid_info + os.sep + bb_name)
    bb = np.round(bb/2)
    bb_sz= bb.shape

    MyDetector = Detector(hm, min_h, min_d, thr)
    hemo_coords = MyDetector.detect()
    
    DiceObj = Dice_metrik(hm,hemo_coords,bb)
    tp,fp,fn = DiceObj.contingencyTab_bboxIndividual()
    
    ######### Zobrazenie
    if tp>0 or fp>0 or fn>0:
        plt.figure()
        plt.imshow(im, cmap='gray')
        plt.plot(hemo_coords[:,1],hemo_coords[:,0],'*r')
        plt.title('TP=' + str(tp) + '  FP=' + str(fp) + '  FN=' + str(fn))
        if bb_sz[1] > 0:
            ax = plt.gca()
                    
            for ii in range(bb_sz[1]):
                rect = patches.Rectangle((bb[0,ii,0],bb[0,ii,1]),
                                          bb[0,ii,2],bb[0,ii,3],
                                          linewidth=1,edgecolor='r',facecolor='none')
                
               
                ax.add_patch(rect)    
            MyDetector = Detector(hm, min_h, min_d, thr)
            hemo_coords = MyDetector.detect()    
            plt.plot(hemo_coords[:,1],hemo_coords[:,0],'*r')
                
        plt.savefig('../detect_results' + os.sep + 'example_image' + im_name + '.png')
        plt.show()
        plt.close()
    
    

    out = [tp, fp, fn]
    return out

def get_dice_val(min_h, min_d, thr):
    save_valid_hm = r"D:\Users\Nemcek\testRes\testHeatmaps"
    save_valid_im = r"D:\Users\Nemcek\testRes\testImages"
    save_valid_info = r"D:\Users\Nemcek\testRes\testInfo"

    hms = os.listdir(save_valid_hm)
    ims = os.listdir(save_valid_im)
    bbs = os.listdir(save_valid_info)
    
    with Pool() as pool:
        out = pool.starmap(Wrapper(min_h,min_d,thr),zip(hms,ims,bbs))
        # print(out)
    
    out = np.array(out)
    tp_fp_fn = np.sum(out, axis=0) 
    # print(tp_fp_fn)
    
    TP = tp_fp_fn[0]
    FP = tp_fp_fn[1]
    FN = tp_fp_fn[2]
    
    dice = 2*TP / (2*TP + FP + FN)
    return dice

class Wrapper(object):
    def __init__(self, min_h, min_d, thr):
        self.min_h = min_h
        self.min_d = min_d
        self.thr = thr
    def  __call__(self, hm_list, im_list, bb_list):
        return get_conting_value(hm_list, im_list, bb_list, self.min_h, self.min_d, self.thr)
        
if __name__ == '__main__':
    
    plt.close('all')
      
    
    min_h = 0.024203
    min_d = 10
    thr = 0.75963
    
 
    dice = get_dice_val(min_h, min_d, thr)
    print(dice)
    
    