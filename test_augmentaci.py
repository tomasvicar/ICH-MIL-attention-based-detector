import torch
import napari
import numpy as np
import SimpleITK as sitk
import random
from scipy.ndimage import rotate
from scipy.ndimage import zoom



from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw



file_name = '../raw_cq/CQ500-CT-1.mhd'



data = read_raw(file_name)


data_orig = data.copy()

size = data.shape

## lr flip
# if random.random()>0.5:
#     data = data[::-1,:,:]



### multiply augmentation
# max_multiplier  = 0.1

# multiplier = 1 + random.random() * max_multiplier
# if random.random()>0.5:
#     multiplier = 1 / multiplier
    
# data = data * multiplier


### add augmentation
# max_add  = 200
# add_value = max_add -2*random.random() 
# data = data + add_value


#all flips and rotations
# if random.random()>0.5:
#     data = data[::-1,:,:]
 
# if random.random()>0.5:
#     data = data[:,::-1,:]

# if random.random()>0.5:
#     data = data[:,:,::-1]


# #####
# data = np.rot90(data,random.randrange(4),axes=(0,1))


# if random.random()>0.5:
#     data = data[::-1,:,:]
 
# if random.random()>0.5:
#     data = data[:,::-1,:]

# if random.random()>0.5:
#     data = data[:,:,::-1]

data = rotate(data, angle=random.randrange(30), axes = (0,1)) 
# data = rotate(data, angle=random.randrange(5), axes = (1,2))

with napari.gui_qt():
    viewer = napari.Viewer(order=[2,1,0])
    viewer.add_image(data,name='data')
    # viewer.add_image(data_orig,name='orig')



