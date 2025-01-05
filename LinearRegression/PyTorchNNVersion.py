import numpy as np
import torch as th
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn

true_w = th.tensor([2, -3.4])
true_b = th.tensor(4.2)

def random_data(w, b, size_of_data):
    X = th.normal(mean = 0, std = 1, size = (size_of_data, len(w)))
    y = th.matmul(X, w) + b
    y += th.normal(mean = 0, std = 0.01, size = y.shape)
    # print(y)

    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    ax[0].set_title('Feature x_1')
    ax[0].scatter(X[:, 0].detach().numpy(), y) # 从图上也可以看出，x_1 和 y 之间的线性关系并没有很明显，这与 w_1 的绝对值更接近 0 相契合
    ax[0].set_xlabel('x_1')
    ax[0].set_ylabel('y')
    ax[1].set_title('Feature x_2')
    ax[1].scatter(X[:, 1].detach().numpy(), y)
    ax[1].set_xlabel('x_2')
    ax[1].set_ylabel('y')
    plt.show()

    return X, y.reshape((-1, 1)) # 将 y 变成列向量

features, labels = random_data(true_w, true_b, 1000)

def load_data(data_arrays, size_of_batch, is_train = True):
    dataset = data.TensorDataset(*data_arrays) # 将数据解包，传入一个 Dataset
    return data.DataLoader(dataset, size_of_batch, shuffle = is_train)

size_of_batch = 10
data_iter = load_data((features, labels), size_of_batch) # 创建一个迭代器


net = nn.Sequential(nn.Linear(2, 1)) # 创建一个线性层
net[0].weight.data.normal_(0, 0.01) # 初始化参数
net[0].bias.data.fill_(0) # 初始化参数

loss = nn.MSELoss() # 损失函数

step = th.optim.SGD(net.parameters(), lr = 0.03) # 小批量随机梯度下降 SGD: 随机梯度下降

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y) # 计算损失
        l.backward() # 反向传播
        step.step() # 更新参数
        step.zero_grad()  # 清空梯度
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    print(f'w: {net[0].weight.data}, b: {net[0].bias.data}')
