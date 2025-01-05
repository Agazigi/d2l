import torch as th
from torch import nn

'''
    卷积是为了提取、寻找特征
    池化是为了压缩数据
    激活是为了加强特征
'''

def corr2d(X, K):
    h, w = K.shape
    Y = th.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = ((X[i: i + h, j: j + w] * K)).sum()
    return Y

def test1():
    X = th.Tensor([[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]])
    K = th.Tensor([[0.0, 1.0],
                   [2.0, 3.0]])
    print(corr2d(X,K))

# test1()

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(th.rand(kernel_size))
        self.bias = nn.Parameter(th.zeros(1))
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

'''
    学习卷积核
'''
def test2():
    conv2d = nn.Conv2d(1,1,kernel_size=(1, 2), bias=False)
    X = th.ones((6, 8))
    X[:, 2:6] = 0
    K = th.Tensor([[1.0, -1.0]]) # 真实的卷积核
    Y = corr2d(X, K) # 得到真实的输出
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2
    for i in range(100):
        Y_hat = conv2d(X) # 预测输出
        l = (Y_hat - Y) ** 2 # 损失函数
        conv2d.zero_grad() # 一定每次反向传播之前要清空梯度
        l.sum().backward() # 反向传播
        conv2d.weight.data[:] -= lr * conv2d.weight.grad # 更新权重
        if (i + 1) % 2 == 0:
            print(f'batch {i+1}, loss {l.sum():.3f}')
    print(conv2d.weight)

# test2()


'''
    填充
'''
def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

def test3():
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = th.rand(size = (8, 8))
    print(comp_conv2d(conv2d, X))


# test3()


'''
    步幅
'''
def test4():
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    X = th.rand(size = (8, 8))
    Y = comp_conv2d(conv2d, X)
    print(Y)
    print(Y.shape)

# test4()
def test5():
    conv2d = nn.Conv2d(1, 1, kernel_size=(3,5), padding=(0,1), stride=(3,4))
    X = th.rand(size = (8, 8))
    Y = comp_conv2d(conv2d, X)
    print(Y)
    print(Y.shape)

# test5()


'''
    多输入通道
'''
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

def test6():
    X = th.Tensor([[[0.0, 1.0, 2.0],
                     [3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0]],
                    [[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]]])
    K = th.Tensor([[[0.0, 1.0],
                    [2.0, 3.0]],
                   [[1.0, 2.0],
                    [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))

# test6()


'''
    多输出通道
'''
def corr2d_multi_in_out(X, K):
    return th.stack([th.Tensor(corr2d_multi_in(X, k)) for k in K], 0)

def test7():
    X = th.Tensor([[[0.0, 1.0, 2.0],
                     [3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0]],
                    [[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]]])
    K = th.Tensor([[[0.0, 1.0],
                   [2.0, 3.0]],
                  [[1.0, 2.0],
                   [3.0, 4.0]]])
    K = th.stack((K, K+1, K+2), 0)
    print(corr2d_multi_in_out(X, K))

# test7()


'''
    1 X 1 卷积层：用来改变通道数
'''
def corr2d_multi_in_out_1x1(X, K):
    c_in, h, w = X.shape
    c_out = K.shape[0]
    X = X.reshape((c_in, h * w))
    K = K.reshape((c_out, c_in))
    Y = th.matmul(K, X)
    return Y.reshape((c_out, h, w))

def test8():
    X = th.normal(0, 1, (3, 3, 3))
    K = th.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    print(Y1)
    Y2 = corr2d_multi_in_out(X, K) # 先前的通用实现
    print(Y2)
    assert float(th.abs(Y1 - Y2).sum()) < 1e-6

test8()
