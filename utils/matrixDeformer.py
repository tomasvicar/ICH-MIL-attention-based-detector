import random


import numpy as np
import cv2



class MatrixDeformer():
    def __init__(self,scale_range=0.05,shears_range=0.05,tilt_range=0.001,translation_range=50,rotation_range=20):
        
        
        self.sx = 1 + scale_range * rand()
        if rand()>0.5:
            self.sx = 1/self.sx
            
        self.sy = 1 + scale_range * rand()
        if rand()>0.5:
            self.sy = 1/self.sy
        
        self.gx=(0-shears_range)+shears_range*2*rand()
        self.gy=(0-shears_range)+shears_range*2*rand()
        
        self.tx=(0-tilt_range)+tilt_range*2*rand()
        self.ty=(0-tilt_range)+tilt_range*2*rand()
        
        self.dx=(0-translation_range)+translation_range*2*rand()
        self.dy=(0-translation_range)+translation_range*2*rand()
        
        self.r=(0-rotation_range)+rotation_range*2*rand()
        
        
        
    def augment(self,img):
        
        (cols,rows) = img.shape
        
        
        M=np.array([[self.sx, self.gx, self.dx], [self.gy, self.sy, self.dy],[self.tx, self.ty, 1]])
        
        R=cv2.getRotationMatrix2D((cols / 2, rows / 2), self.r, 1)
        R=np.concatenate((R,np.array([[0,0,1]])),axis=0)
        
        matrix= np.matmul(R,M)
        
        
        flags=cv2.INTER_LINEAR
#        cv2.BORDER_CONSTANT
        img = cv2.warpPerspective(img,matrix, (cols,rows),flags=flags,borderMode=cv2.BORDER_REPLICATE)
        
        
        return img
    

    
def rand():
    return random.random()
    