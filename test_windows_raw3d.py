import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw


file_name = '../raw_cq_subsampled4x/CQ500-CT-1.mhd'



data = read_raw(file_name)

data = data.astype(np.float32).copy()
        



plt.hist(data.flatten(),255)
