import numpy as np
import SimpleITK as sitk
import napari
import matplotlib.pyplot as plt

from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw


file_name = '../raw_cq_subsampled4x/CQ500-CT-1.mhd'



data = read_raw(file_name)

data = data.astype(np.float32).copy()
        
        
##### Subdural window
data_sd = data.copy()
data_sd = ((data_sd-1009.0)/(1139.0-1009.0)) * (2**12)
data_sd[data_sd<0.0] = 0.0
data_sd[data_sd>2**12] = 2**12
##### Brain window
data_br = data.copy()
data_br = ((data_br-1024.0)/(1104.0-1024.0)) * (2**12)
data_br[data_br<0.0] = 0.0
data_br[data_br>2**12] = 2**12



print(np.mean(data))
print(np.std(data))
print(np.mean(data_sd))
print(np.std(data_sd))
print(np.mean(data_br))
print(np.std(data_br))

