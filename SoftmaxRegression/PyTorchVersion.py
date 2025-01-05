import multiprocessing
import torch as th
import torchvision as thv
import torchvision.transforms as transforms
from torch.utils import data
import matplotlib.pyplot as plt
import myutils

def get_dataloader_workers():
    return 4

def get_fashion_mnist_labels(labels):
    """
    :param labels: 传入的数字标签
    :return: 返回标签对应的文字
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

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

batch_size = 256 # 设置批量大小

num_inputs = 784 # 一个图片的大小 28 * 28
num_outputs = 10 # 一共 10 种类别

lr = 0.1

def softmax(X):
    """
        这个函数将一个输入的矩阵进行 softmax 转换
    """
    X_exp = th.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp / partition

def net(X):
    """
        这是我们的模型:
    """
    return softmax(th.matmul(X.reshape(-1, W.shape[0]), W) + b) # -1 表示自动计算

def sgd(params, lr, batch_size):
    with th.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def updater(batch_size): # 返回一个函数，用来更新参数
    return sgd([W, b], lr, batch_size)

def loss(y_hat, y):
    return -th.log(y_hat[range(len(y_hat)), y]) 

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, th.nn.Module): # 判断是否是 pytorch 的 Module
        net.eval() # 设置为评估模式
    metric = myutils.Accumulator(2) # 统计正确预测数和总预测数
    with th.no_grad(): # 评估模式下不进行梯度计算
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, th.nn.Module):
        net.train() # 设置为训练模式
    metric = myutils.Accumulator(3) # 统计训练损失、训练准确数、样本数
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

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = myutils.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                                legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert 1 >= train_acc > 0.7, train_acc
    # assert 1 >= test_acc > 0.7, test_acc

num_epochs = 10

def show_images(imgs, num_rows, num_cols, titles=None, scale=3):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if th.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def predict(net, test_iter, n=6):
    for X, y in test_iter:
        trues = get_fashion_mnist_labels(y)
        predictions = get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true +'\n' + pred for true, pred in zip(trues, predictions)]
        show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        # break

if __name__ == '__main__':
    multiprocessing.freeze_support() # 运行时需要加上这句，支持多进程

    # 获取模型参数
    W = th.normal(mean=0, std=0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 注意 size 的写法
    b = th.zeros(num_outputs, requires_grad=True)
    train_iter, test_iter = load_mnist(batch_size)

    # print(evaluate_accuracy(net, test_iter))
    train(net, train_iter, test_iter, loss, num_epochs, updater)
    predict(net, test_iter)
    plt.show()


