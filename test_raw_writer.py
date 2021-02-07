import numpy as np
import napari

from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw
from utils.raw_loaders import write_raw


file_name = '../Brain_VFN_0004.mhd'
file_name2 =  '../Brain_VFN_0004_2.mhd'


img1 = read_raw(file_name)
img1 = img1.astype(np.float32)
write_raw(img1,file_name2)

img2 = read_raw(file_name2)



print(np.sum(np.abs(img1-img2)))


with napari.gui_qt():
    viewer = napari.Viewer(order=[2,1,0])
    viewer.add_image(img1,name='img1')
    viewer.add_image(img2,name='img2')





