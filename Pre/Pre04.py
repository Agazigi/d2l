import os
import pandas as pd
import torch as th

'''
    Pandas 处理
'''

def init():
    global flag
    os.makedirs(os.path.join('../../HandsOnDeepLearning', 'data'), exist_ok=True)
    data_file = os.path.join('../../HandsOnDeepLearning', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n') # 列名
        f.write('NA,Pave,127500\n') # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    data = pd.read_csv(data_file)
    print(data)
    #
    # inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    # inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
    # print(inputs)
    #
    # inputs = pd.get_dummies(inputs, dummy_na=True)
    #
    # print(inputs)
    #
    # print(outputs)
    #
    # X = th.tensor(inputs.to_numpy(dtype=float))
    # y = th.tensor(outputs.to_numpy(dtype=float))
    # print(X)
    # print(y)



    cnt_max = 0
    for i in data.head():
        cnt = data[i].isna().sum()
        if cnt > cnt_max:
            cnt_max = cnt
            flag = i
    new_data = data.drop(flag, axis=1)
    print(new_data)

    new_data = new_data.fillna(new_data.mean())
    print(new_data)
    new_data = th.tensor(new_data.to_numpy(dtype=float))
    print(new_data)


init()