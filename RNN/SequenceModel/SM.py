import torch as th
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
from torch.utils import data
import myutils

T = 1000 # 创建 1000 个点
time = th.arange(1, T + 1, dtype=th.float32) # 创建时间序列
x = th.sin(0.01 * time) + th.normal(0, 0.2, (T,)) # 创建正弦函数 + 噪声

def draw_data():
    global time, x
    plt.plot(time,x)
    plt.xlabel('time')
    plt.ylabel('x')
    plt.xlim([1, 1000])
    plt.grid()
    plt.show()

# draw_data()

tau = 4 # 向前取 4 个历史值
batch_size = 16 # 小批量
n_train = 600 # 用于训练

features = th.zeros((T - tau, tau)) # 创建特征矩阵
for i in range(tau):
    features[:, i] = x[i: T - tau + i] # 填充每个特征的值，即之前四个的值
labels = x[tau:].reshape((-1, 1)) # 填充标签，即下一个值，跳过前面四个值

loss = th.nn.MSELoss(reduction='none') # 使用 MSELoss

dataset = data.TensorDataset(*(features[:n_train], labels[:n_train])) # 将特征和标签组合成一个数据集
train_iter = data.DataLoader(dataset, batch_size, shuffle=True) # 创建小批量数据

def init_weights(m): # 初始化参数
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)

def get_net(): # MLP
    net = th.nn.Sequential(
        th.nn.Linear(4, 10),
        th.nn.ReLU(),
        th.nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net

net = get_net()


def evaluate_loss(net, data_iter, loss):
    reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

    metric = myutils.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = reshape(y, out.shape)
        l = loss(out, y)
        metric.add(reduce_sum(l), size(l))
    return metric[0] / metric[1]

def train(net, loss, train_iter, epochs, lr):
    updater = th.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        print('epoch %d, loss %f' % (epoch + 1, evaluate_loss(net, train_iter, loss)))

train(net, loss, train_iter, epochs=5, lr=0.01)

onestep = net(features)

# plt.plot(time,x.detach().numpy(), 'b-')
# plt.plot(time[tau:], onestep.detach().numpy(), 'r-')
# plt.xlabel('time')
# plt.ylabel('x')
# plt.figure(1,(6, 3))
# plt.xlim(1, 1000)
# plt.grid()
# plt.legend(['data', '1-step pred'], loc='upper left')
# plt.show()


multistep_predictions = th.zeros(T) # 创建预测序列
multistep_predictions[:n_train+tau] = x[:n_train+tau] # 填充训练数据
for i in range(n_train+tau, T): # 预测后面的数据,我们使用的是前边的 tau 个值
    multistep_predictions[i] = net(multistep_predictions[i-tau:i].reshape((1,-1)))
plt.plot(time,x.detach().numpy(), 'b-')
plt.plot(time[tau:], onestep.detach().numpy(), 'r-')
plt.plot(time[n_train+tau:], multistep_predictions[n_train+tau:].detach().numpy())
plt.xlabel('time')
plt.ylabel('x')
plt.figure(1,(6, 3))
plt.xlim(1, 1000)
plt.grid()
plt.legend(['data', '1-step pred', 'multistep pred'], loc='upper left')
plt.show()