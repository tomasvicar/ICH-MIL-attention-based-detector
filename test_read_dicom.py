import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils.load_dicom_slice import load_dicom_slice
from utils.get_dicom_slice_size import get_dicom_slice_size


name = r'D:\vicar\kuba_embc2021\Data3_CQ500\CQ500CT0 CQ500CT0\Unknown Study\CT PLAIN THIN\CT000000.dcm'


size = get_dicom_slice_size(name)


data = load_dicom_slice(name)



plt.hist(data.flatten(),255)