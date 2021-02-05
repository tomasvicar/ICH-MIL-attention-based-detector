import numpy as np


from utils.raw_loaders import get_size_raw
from utils.raw_loaders import read_raw



file_name = '../Brain_VFN_0004.mhd'



import time


start = time.time()
size = get_size_raw(file_name)
end = time.time()
print(end - start)


start = time.time()
img1 = read_raw(file_name)
end = time.time()
print(end - start)




start = time.time()
img2 = read_raw(file_name,extract_size=[100,100,5],current_index=[5,6,7])
end = time.time()
print(end - start)


img3 = img1[5:5+100,6:6+100,7:7+5]

print(np.sum(np.abs(img3-img2)))

print(np.std(img1))
print(np.mean(img1))


import pandas as pd

dfs = pd.read_csv('../VFN_Annotations.csv',delimiter=';')



