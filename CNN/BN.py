import torch as th
from torch import nn
import myutils
from torchvision import transforms
import torchvision as thv
from torch.utils import data
import matplotlib.pyplot as plt

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not th.is_grad_enabled(): # 当前是否是训练模式： 否
        X_hat = (X - moving_mean) / th.sqrt(moving_var + eps)
    else: # 当前是训练模式： 是
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2: # 当前是全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else: # 当前是卷积层
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / th.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean # 更新移动平均值
        moving_var = momentum * moving_var + (1.0 - momentum) * var # 用来平滑历史的统计量
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(th.ones(shape))
        self.beta = nn.Parameter(th.zeros(shape))
        self.moving_mean = th.zeros(shape)
        self.moving_var = th.ones(shape)
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4, 120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

lr = 1
num_epochs = 20
batch_size = 256

def get_dataloader_workers():
    return 4

def load_mnist(batch_size, resize=None):
    """下载数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize: # 缩放
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # 组合转换操作
    mnist_train = thv.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = thv.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

train_iter, test_iter = load_mnist(batch_size)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = myutils.Accumulator(2)
    with th.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = th.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = myutils.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = myutils.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = myutils.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with th.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    plt.show()

def try_gpu(i=0):
    """
    用来获取GPU，没有就返回CPU，可以指定GPU的编号
    """
    if th.cuda.device_count() >= i + 1:
        return th.device(f'cuda:{i}')
    return th.device('cpu')

train(net, train_iter, test_iter, num_epochs=20, lr=0.9, device=try_gpu())
