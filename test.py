import cupy as cp
import time

import numpy as np
import pandas as pd

origin_df = pd.read_csv("D:\PythonProjects\data_final\data\processed\geolife_data.csv")
clean_df = pd.read_csv("D:\PythonProjects\data_final\data\processed\geolife_beijing_cleaned_data.csv")

# 打印行数
print("origin_df.shape: ", origin_df.shape)
print("clean_df.shape: ", clean_df.shape)