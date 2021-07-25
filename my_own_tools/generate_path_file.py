'''
generate .csv file for e2vid code.
train.csv, valid.csv include all 
'''

from math import inf
import os
import pandas as pd
import json

# data_path = '/home/zhujun/Documents/data/e2vid/'
# data_path = '/home/zhujun/Documents/data/e2vid/data_paper/' # 论文数据
data_path = '/home/zhujun/Documents/data/e2vid/data_paper_firenet/'
config_path = '/home/zhujun/Documents/lenovo_dvs/event_cnn_minimal/config/'

max_num = inf

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
# name = 'reconstruction'
name = 'reconstruction_firenet'
with open(config_path+name+'.json','r') as f:
        params = json.load(f)
        params['data_loader']['args']['data_file'] = train_path + '.csv'
        params['valid_data_loader']['args']['data_file'] = valid_path + '.csv'
        params['trainer']['save_dir'] = data_path + 'model'
with open(config_path+name+'.json','w') as f:
        json.dump(params, f, indent=4)
# import copy
# a = [1, 2, 3, 4, ['a', 'b']] #原始对象

# b = a                       #赋值，传对象的引用
# c = copy.copy(a)            #对象拷贝，浅拷贝
# d = copy.deepcopy(a)        #对象拷贝，深拷贝
 
# a.append(5)                 #修改对象a
# a[4].append('c')            #修改对象a中的['a', 'b']数组对象
# >>> a
# [1, 2, 3, 4, ['a', 'b', 'c'], 5]
# >>> b
# [1, 2, 3, 4, ['a', 'b', 'c'], 5]
# >>> c
# [1, 2, 3, 4, ['a', 'b', 'c']]
# >>> d
# [1, 2, 3, 4, ['a', 'b']]

# >>> a1 = a[1] # a1 指向a的不可变子对象
# >>> a4 = a[4] # a4指向a的可变子对象
# >>> a1 = 10
# >>> a4[0] = 'ddd'
# >>> a
# [1, 2, 3, 4, ['ddd', 'b', 'c'], 5] # 不可变子对象没变，而可变子对象改变！！！！
# >>> b
# [1, 2, 3, 4, ['ddd', 'b', 'c'], 5]
# >>> c
# [1, 2, 3, 4, ['ddd', 'b', 'c']]
# >>> d
# [1, 2, 3, 4, ['a', 'b']]

# C/C++：变量对应内存中的一块区域，当修改这个值时，直接修改内存区域中的值。
# Python中的变量都是指针，这确实和之前学过的强类型语言是有不同的。
# 因为变量是指针，所以所有的变量无类型限制，可以指向任意对象。指针的内存空间大小是与类型无关的，其内存空间只是保存了所指向数据的内存地址。

# >>> a1=a[1] # a1指向了不可变对象
# >>> id(a1)
# 93837811561248
# >>> a1 = 10 # 这个操作相当于直接给a1换了个指向的对象，所以地址当然不同
# >>> id(a1)
# 93837811561504

# >>> id(a4)
# 140125087259856
# >>> a4[0]='aaaa' # 这个操作实在a4指向的地址上修改，因此
# >>> a4
# ['aaaa', 'b', 'c'] # a4指向的是列表（可变对象），可变对象的值可以动态修改
# >>> id(a4)
# 140125087259856

# 不可变对象一旦创建，其内存中存储的值就不可以再修改了。
# 如果想修改，只能创建一个新的对象，然后让变量指向新的对象，所以前后的地址会发生改变。而可变对象在创建之后，其存储的值可以动态修改。
# >>> a = 666
# >>> id(a)1365442984464
# >>> a += 1
# >>> id(a)1365444032848