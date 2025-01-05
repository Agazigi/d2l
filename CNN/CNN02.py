import torch as th
from torch import nn

'''
    pooling：池化层 max pooling, avg pooling
'''
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = th.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

def test1():
    X = th.tensor([[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))

# test1()

'''
    池化层的填充与步幅    
'''
def test2():
    X = th.arange(16,dtype=th.float32).reshape((1,1,4,4))
    pool2d = nn.MaxPool2d(3) # 使用 pytorch 的池化层，默认情况下步幅也是 (3, 3)
    print(pool2d(X))

# test2()

def test3():
    pool2d = nn.MaxPool2d((2,3), stride=(2,3), padding=(0,1))
    X = th.arange(16,dtype=th.float32).reshape((1,1,4,4))
    print(pool2d(X))

# test3()
def test4():
    X = th.arange(16,dtype=th.float32).reshape((1,1,4,4))
    X = th.cat((X, X + 1), 1)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))

test4()