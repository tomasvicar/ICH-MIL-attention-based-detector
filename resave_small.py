from glob import glob
import numpy as np
import os
from scipy.ndimage import zoom

from utils.raw_loaders import read_raw
from utils.raw_loaders import write_raw

factor = 0.25

src_path = '../raw_cq'
dst_path = '../raw_cq_subsampled4x'


if not os.path.exists(dst_path):
    os.makedirs(dst_path)

file_names = glob(src_path + '/*.mhd')


for file_num,file_name in enumerate(file_names):
    print(str(file_num) + '/' + str(len(file_names)))
    
    img1 = read_raw(file_name)
    img1 = img1.astype(np.float32)
    img1 = zoom(img1,[factor,factor,1])
    write_raw(img1,file_name.replace(src_path,dst_path))

    
    











