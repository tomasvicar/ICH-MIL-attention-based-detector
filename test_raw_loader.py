import numpy as np
import SimpleITK as sitk
import napari

from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw



file_name = '../raw_cq/CQ500-CT-1.mhd'



tmp = read_raw(file_name)


with napari.gui_qt():
    viewer = napari.Viewer(order=[2,1,0])
    viewer.add_image(tmp,name='img')




