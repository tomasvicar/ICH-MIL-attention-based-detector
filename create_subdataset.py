import pandas as pd
import numpy as np
from shutil import copyfile

src_path = 'D:\Jakubicek\RSNA\data\RSNA_sub'
dst_path = 'D:\Jakubicek\RSNA\data\RSNA_sub_sample'

src_table_name = 'label_table_dicomcol_merge.csv'
dst_table_name = 'label_table_dicomcol_merge_sample.csv'

df = pd.read_csv(src_table_name,delimiter=';')






newdf = df[df.PacNum.isin(range(20))]

newdf.to_csv(dst_table_name,sep=';')



IDs = newdf.ID.tolist()

for ID in IDs:
    
    src = src_path + '/' + ID + '.dcm'
    
    dst = dst_path + '/' + ID + '.dcm'
    
    
    copyfile(src, dst)
    
    
    
    
    