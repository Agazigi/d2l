import math

import matplotlib.pyplot as plt
import torch as th
from torch import nn
import numpy as np
from myutils import Accumulator
from myutils import Animator
from torch.utils import data

max_degree = 20 # 多项式的最大阶数
n_train, n_test = 100, 100 # 训练和测试数据量
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 4个多项式项

features = np.random.normal(size = (n_train + n_test, 1)) # 随机生成n_train+n_test个1维数据 x
np.random.shuffle(features) # 随机打乱数据 x1, x2, ... x20
poly_features = np.power(features, np.arange(max_degree).reshape(1,-1)) # 生成多项式项 x1^0, x2^1, ..., x20^20
# print(poly_features)
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1) # gamma(i) = (i-1)!

labels = np.dot(poly_features, true_w) # w1 * x1^0 + w2 * x2^1 + ... + w20 * x20^20 + noise
labels += np.random.normal(scale=0.1, size=labels.shape) # noise

# 转换为张量
true_w, features, poly_features, labels = [th.tensor(x, dtype=th.float32) for x in [true_w, features, poly_features, labels]]



def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter: # 遍历数据集
        out = net(X) # 模型输出
        y = y.reshape(out.shape) # 将标签 reshape 成和输出一样的形状
        l = loss(out, y) # 计算损失
        metric.add(l.sum(), l.numel()) # 累加损失和样本数
    return metric[0] / metric[1] # 返回平均损失

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, th.nn.Module):
        net.train() # 设置为训练模式
    metric = Accumulator(3) # 统计训练损失、训练准确数、样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, th.optim.Optimizer): # PyTorch 内置的 optimizer
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1] # 输入维度
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0]) # 批量大小
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    trainer = th.optim.SGD([
        {"params": net[0].weight, "weight_decay": 2}, # weight_decay是L2正则项
    ], lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
    plt.show()

if __name__ == "__main__":
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
