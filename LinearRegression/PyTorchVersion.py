import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th

# 我们定义真实的模型参数
true_w = th.tensor([3, -7.5])
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

# 获取训练数据
features, labels = random_data(true_w, true_b, 1000)
# print(features)
# print(labels)

def data_iteration(step_size, features, labels):
    size_of_data = len(features) # 总的数据量
    data_num_list = list(range(size_of_data))
    random.shuffle(data_num_list) # 随机打乱数据
    for i in range(0, size_of_data, step_size):
        batch_indices = th.tensor(data_num_list[i: i + step_size]) # 随机选取 batch_size 个数据
        yield features[batch_indices], labels[batch_indices] # 迭代，下次 yield 会从这里开始

# for X, y in data_iteration(10, features, labels):
#     print(X, '\n', y)

init_w = th.normal(mean = 0, std = 0.01, size = (2, 1), requires_grad = True)
init_b = th.zeros(size = [1], requires_grad = True) # 打开梯度计算
alpha = 0.01

# print(init_w, init_b)
def linear_model(X, w, b):
    return th.matmul(X, w) + b

def cost_function(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def update_params(params, alpha, size_of_data):
    with th.no_grad(): # 禁止梯度计算 no_grad() 上下文管理器确保在该块代码中执行的所有操作不会记录计算图，从而节省内存并加速推理过程。
        for param in params:
            param -= alpha * param.grad / size_of_data # 更新参数，梯度下降
            param.grad.zero_() # 清空梯度

network = linear_model
loss = cost_function
num_iteration = 100
size_of_data = 10

def gradient_descent():
    for it in range(num_iteration):
        for X, y in data_iteration(10, features, labels):
            l = loss(network(X, init_w, init_b), y)
            l.sum().backward() # 反向传播
            update_params([init_w, init_b], alpha, size_of_data)
        with th.no_grad():
            print(f'iteration: {it}, w: {init_w}, b: {init_b}')
            print(f'loss: {loss(network(features, init_w, init_b), labels).mean()}')

gradient_descent()