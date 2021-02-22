import numpy as np
import pandas as pd
import os

from config import Config





df = pd.read_csv(Config.data_table_path,delimiter=';')


file_names = df['ID'].tolist()
Pat_ind = df['PacNum'].to_numpy()
lbl = df['Label'].to_numpy()

file_names = [Config.data_path + os.sep + file_name + '.dcm' for file_name in file_names]









