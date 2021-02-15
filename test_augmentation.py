import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils.load_dicom_slice import load_dicom_slice
from utils.get_dicom_slice_size import get_dicom_slice_size

from my_dataset import MyDataset


name = r'D:\vicar\kuba_embc2021\Data3_CQ500\CQ500CT0 CQ500CT0\Unknown Study\CT PLAIN THIN\CT000010.dcm'


size = get_dicom_slice_size(name)


data = load_dicom_slice(name).astype(np.float32)



plt.imshow(data)
plt.show()



data = MyDataset.data_augmentation(data)


plt.imshow(data)
plt.show()


data = MyDataset.data_tranform(data)



plt.imshow(data[0,:,:])
plt.show()


plt.imshow(data[1,:,:])
plt.show()


plt.imshow(data[2,:,:])
plt.show()









