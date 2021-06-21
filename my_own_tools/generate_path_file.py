
'''
generate .csv file for e2vid code.
train.csv, valid.csv include all 

'''

import os
import pandas as pd

train_path='/home/larrydong/dataset/e2vid/train'
valid_path='/home/larrydong/dataset/e2vid/valid'

L = []
for root, dirs, files in os.walk(train_path):
    for file in files:  # 遍历所有文件名
        if os.path.splitext(file)[1] == '.h5':  
            L.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表
df = pd.DataFrame.from_dict(L)
df.to_csv('train.csv', header=None, index=None)

L = []
for root, dirs, files in os.walk(valid_path):
    for file in files:  # 遍历所有文件名
        if os.path.splitext(file)[1] == '.h5':  
            L.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表
df = pd.DataFrame.from_dict(L)
df.to_csv('valid.csv', header=None, index=None)
