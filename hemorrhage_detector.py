import numpy as np
import os
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import json
import pickle
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.measure import label
from skimage import color


class Detector():
    
    def __init__(self, heatmap, minH, minDist, thr ):
        self.heatmap = heatmap
        self.minH = minH
        self.minDist = minDist
        self.thr = thr
    
    def detect(self):
        p1 = peak_local_max(self.heatmap, min_distance=self.minDist,
                            threshold_abs=self.thr)
        if self.minH > 0:
            p2=h_maxima(self.heatmap,self.minH)
            
            delVec = []
            for it,p in enumerate(p1):
                if p2[int(p[0]),int(p[1])] == 0:
                    delVec.append(it)
            p1 = np.delete(p1, delVec, axis=0)  
        
        
        return p1
    
class Dice_metrik():
    def __init__(self, heatmap, points, BBs):
        self.heatmap = heatmap
        self.points = points
        self.BBs = BBs
        

        
    def contingencyTab(self):
        
        TP = 0
        FP = 0
        FN = 0
        bb = self.BBs
        
        BBarea = np.zeros(self.heatmap.shape)
        if self.BBs.shape[1] > 0:
            for ii in range(self.BBs.shape[1]):
                BBarea[ int(bb[0,ii,1]):int(bb[0,ii,1]+bb[0,ii,3]),
                        int(bb[0,ii,0]):int(bb[0,ii,0]+bb[0,ii,2]) ] = np.ones((int(bb[0,ii,3]),int(bb[0,ii,2])))
            
        
        if self.points.shape[0] > 0 :  
            for p in self.points:
                if BBarea[int(p[0]),int(p[1])] == 1:
                    TP = TP + 1
                elif  BBarea[int(p[0]),int(p[1])] == 0:
                    FP = FP + 1
        else:
            if self.BBs.shape[1] > 0:
                FN = FN + 1
            
        if TP > 0:
            TP = 1
            
        return TP, FP, FN
    
    
    def contingencyTab_bboxIndividual(self):
        
        TP = 0
        FP = 0
        FN = 0
        bb = self.BBs
        
        
        ctrl_points = np.zeros((1,self.points.shape[0])) 
       
        if self.BBs.shape[1] > 0:
            for ii in range(self.BBs.shape[1]):
                BBarea = np.zeros(self.heatmap.shape)
                BBarea[ int(bb[0,ii,1]):int(bb[0,ii,1]+bb[0,ii,3]),
                            int(bb[0,ii,0]):int(bb[0,ii,0]+bb[0,ii,2]) ] = np.ones((int(bb[0,ii,3]),int(bb[0,ii,2])))
                any_point = 0
                if self.points.shape[0] > 0 :                                     
                    for it in range(self.points.shape[0]):
                        p = self.points[it]
                        if BBarea[int(p[0]),int(p[1])] == 1:
                            any_point += 1
                            ctrl_points[0,it] = 1
                if any_point > 0:
                    TP += 1
                else:
                    FN += 1     
        
        FP = self.points.shape[0] - int(np.sum(ctrl_points, axis=1).astype(int))
                     
        return TP, FP, FN
    
    
    def dice_metrik(self, TP, FP, FN):
        dice = 2*TP / (2*TP + FP + FN)
        return dice
                