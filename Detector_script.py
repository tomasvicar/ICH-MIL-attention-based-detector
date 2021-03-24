import numpy as np
import torch
from torch.utils import data
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



saveImHMpath = r"D:\nemcek\EMBC2021\test_Imgs_Heatmaps_detector"

k = 125

im = np.load(saveImHMpath+ '\img_'+str(k)+'.npy')
hm = np.load(saveImHMpath+ '\hm_'+str(k)+'.npy')


plt.figure()
plt.imshow(im, cmap='gray')
plt.show()

# plt.figure()
# plt.imshow(hm, cmap='gray')
# plt.show()ŕ

min_h = 0.01
min_d = 10
thr = -1
p1 = peak_local_max(hm, min_distance=min_d, threshold_abs=thr)
p2=h_maxima(hm,min_h)

out=np.zeros(hm.shape)
for p in p1:
    out[int(p[0]),int(p[1])]=p2[int(p[0]),int(p[1])]


plt.figure()
plt.imshow(hm, cmap='gray')
plt.plot(p1[:,1],p1[:,0],'*r')
plt.show()


label_out = label(out)
overlay = color.label2rgb(label_out, hm, alpha=0.7, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])

plt.figure()
plt.imshow(overlay, cmap='gray')
plt.show()
