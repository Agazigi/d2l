import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 从 ex1data.txt 中读取数据，并指定列名
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
x_train, y_train = data['population'], data['profit']

# 2. 画出数据的散点图
''' 方式一'''
data.plot(kind = 'scatter', x = 'population', y = 'profit', title = 'Data Scatter', label = 'data point')
plt.show()


''' 方式二
x_train = data['population']
y_train = data['profit']
plt.scatter(x_train, y_train, label='data point')
plt.legend() # 显示图例
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Data Scatter')
plt.show()
'''

alpha = 0.001
w_init = 0
b_init = 0
num_iters = 200

def compute_cost(x, y, w, b):
    """
    计算损失函数
    :param x: 输入向量
    :param y: 输出向量
    :param w: 权重
    :param b: 偏置
    :return: 损失值
    """
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def step_gradient(w, b, alpha, x, y):
    """
    计算梯度并更新参数
    :param w: 权重
    :param b: 偏置
    :param alpha: 学习率
    :param x: 输入向量
    :param y: 输出向量
    :return: 更新后的 w 和 b
    """
    m = x.shape[0]
    df_dw = 0
    df_dj = 0
    for i in range(m):
        f_wb = w * x[i] + b
        df_dw += (f_wb - y[i]) * x[i]
        df_dj += (f_wb - y[i])
    w = w - alpha * df_dw / m
    b = b - alpha * df_dj / m
    return w, b

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    cost_history = []
    w_history = []
    b_history = []

    for i in range(num_iters):
        w_history.append(w)
        b_history.append(b)
        cost_history.append(compute_cost(x, y, w, b))
        w, b = step_gradient(w, b, alpha, x, y)
        if i % 10 == 0:
            print(f'iteration: {i}, cost: {cost_history[-1]}')
            print(f'w: {w}, b: {b}')

    w_range = np.linspace(0, 1, 100)
    b_range = np.linspace(-0.1, 0.1, 100)
    W, B = np.meshgrid(w_range, b_range)

    cost_history = np.array(cost_history)
    w_history = np.array(w_history)
    b_history = np.array(b_history)
    # 计算每个(w, b)组合的成本
    Costs = np.array([[compute_cost(x_train, y_train,w, b) for w in w_range] for b in b_range]) # 注意 w 和 b 的顺序
    # Costs = np.reshape(Costs, shape=(100, 100))
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w_history, b_history, cost_history, color='red', s=100)


    # 绘制曲面
    ax.plot_surface(W, B, Costs, cmap='viridis', alpha=0.5)

    # 设置标签
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Cost')

    # 显示图形
    plt.show()


     # 画出 J(w,b) 和 w,b 的三维曲面
    cost_history = np.array(cost_history)
    w_history = np.array(w_history)
    b_history = np.array(b_history)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w_history, b_history, cost_history)

    # 设置标签
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('J(w, b)')

    # 显示图形
    plt.show()



    plt.plot(range(num_iters), cost_history, label='cost', color='red')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Cost History')
    plt.show()
    plt.plot(range(num_iters), w_history, label='w', color='blue')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('w')
    plt.title('W History')
    plt.show()
    plt.plot(range(num_iters), b_history, label='b', color='green')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('b')
    plt.title('B History')
    plt.show()



    plt.scatter(x, y, label='data point')
    plt.plot(x, w * x + b, label='regression line', color = 'red')  # 画出拟合直线

    x_in = 17.5
    y_predict = w * x_in + b
    plt.scatter(x_in, y_predict, marker='x', color='green', label='predict', s=100)
    print(f'predict: {y_predict}')

    plt.legend()  # 显示图例
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.title('Data Scatter')

    plt.show()

    plt.plot(range(num_iters), cost_history, label='cost', color='red')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Cost History')
    plt.show()


gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters)