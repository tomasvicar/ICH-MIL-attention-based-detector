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


### Random crop
dat_size = data.shape
max_perc = 0.1 
x_max_crop = round(dat_size[0]*max_perc)
y_max_crop = round(dat_size[1]*max_perc)
z_max_crop = round(dat_size[2]*max_perc)

Xc = random.randint(0,x_max_crop)
Yc = random.randint(0,y_max_crop)
Zc = random.randint(0,z_max_crop)

data = data[Xc : int(dat_size[0]-x_max_crop+Xc),
            Yc : int(dat_size[1]-y_max_crop+Yc),
            Zc : int(dat_size[2]-z_max_crop+Zc)]

#### Random resize
max_resize_perc = 0.1 
x_res = random.uniform(1-max_resize_perc,1+max_resize_perc)
y_res = random.uniform(1-max_resize_perc,1+max_resize_perc)
z_res = random.uniform(1-max_resize_perc,1+max_resize_perc)

data = zoom(data, (x_res,y_res,z_res))

with napari.gui_qt():
    viewer = napari.Viewer(order=[2,1,0])
    viewer.add_image(data,name='data')
    # viewer.add_image(data_orig,name='orig')



