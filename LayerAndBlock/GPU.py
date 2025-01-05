import torch as th
from torch import nn

# th.device('cpu')
# 默认张量是在CPU上计算
# th.device('cuda') 默认GPU
# print(th.cuda.device_count())
'''
    此电脑一个 GPU
'''
def try_gpu(i=0):
    """
    用来获取GPU，没有就返回CPU，可以指定GPU的编号
    """
    if th.cuda.device_count() >= i + 1:
        return th.device(f'cuda:{i}')
    return th.device('cpu')

def try_all_gpus():
    """
    返回所有GPU，没有就返回CPU
    """
    devices = [th.device(f'cuda:{i}') for i in range(th.cuda.device_count())]
    return devices if devices else [th.device('cpu')]

# print(try_gpu())
# print(try_all_gpus())

# x = th.Tensor([1, 2, 3])
# print(x.device)
'''
    要进行操作的两个数需要在同一个设备上，否则会报错。
'''

# X = th.ones(2,3,device=try_gpu())
# Y = th.ones(3,2,device=try_gpu(1))
import time as T
# T.sleep(30) 可以在 nvidia-smi 中查看
# Z = X @ Y  报错
# print(Z)
# Z = X.cuda(1)
# print(X @ Y)

net = nn.Sequential(nn.Linear(3, 1))
net.to(device=try_gpu())
# print(net.state_dict())


