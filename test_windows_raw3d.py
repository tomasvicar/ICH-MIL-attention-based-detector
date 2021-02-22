import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw


# file_name = r"D:\vicar\kuba_embc2021\Data3_CQ500\CQ500CT0 CQ500CT0\Unknown Study\CT Plain\CT000019.dcm"
file_name = r"D:\vicar\kuba_embc2021\RSNA_sub_sample\ID_3cf26f6e9.dcm"

data = read_raw(file_name)

data = data.astype(np.float32).copy()
        



data = data
data[data<-500] = 0
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



data = np.concatenate((np.expand_dims(data, axis=0),
                        np.expand_dims(data_br, axis=0),
                        np.expand_dims(data_sd, axis=0)), axis=0)




plt.imshow(data_br)
plt.show()