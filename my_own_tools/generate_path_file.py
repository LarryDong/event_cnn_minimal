
'''
generate .csv file for e2vid code.
train.csv, valid.csv include all 

'''

import os
import pandas as pd
import json

data_path = '/home/zhujun/Documents/data/e2vid/'
config_path = '/home/zhujun/Documents/lenovo_dvs/event_cnn_minimal/config/'

max_num = 10

# 生成csv
train_path= data_path + 'train'
valid_path= data_path + 'valid'
for path in [train_path, valid_path]:
    L = []
    for root, dirs, files in os.walk(path):
        for file in files:  # 遍历所有文件名
            if os.path.splitext(file)[1] == '.h5':  
                L.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表
            if len(L)>max_num: # 限制最大数量
                break
    df = pd.DataFrame.from_dict(L)
    df.to_csv(path+'.csv', header=None, index=None)

# 修改config
name = 'reconstruction'
with open(config_path+name+'.json','r') as f:
        params = json.load(f)
        params['data_loader']['args']['data_file'] = train_path + '.csv'
        params['valid_data_loader']['args']['data_file'] = valid_path + '.csv'
        params['trainer']['save_dir'] = data_path + 'model'
with open(config_path+name+'.json','w') as f:
        json.dump(params, f, indent=4)