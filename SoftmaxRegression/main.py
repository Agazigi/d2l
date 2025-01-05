import torch as th
import torchvision as thv
from torch.utils import data
from torchvision import transforms # 用来转换数据
import matplotlib.pyplot as plt

# 使用框架内置的函数 torchvision.datasets.FashionMNIST 读取数据，并使用 transforms.ToTensor() 转换为张量
# mnist_train = thv.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# mnist_test = thv.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

'''
    Fashion‐MNIST由10个类别的图像组成，每个类别由训练数据集（train dataset）中的6000张图像和测试数据
    集（test dataset）中的1000张图像组成。
'''
# print(mnist_train)
# print(mnist_test)
#
# print(len(mnist_train))
# print(len(mnist_test))

# print(mnist_train[0][0].shape) # (1, 28, 28)， 其中  1 表示单通道， [0][0] 是图像， [0][1] 是标签
# 第一个 [] 的索引表示第几张图片，第二个 [] 的索引 其中 0 表示图片，1 表示标签
# print(mnist_train[0][1]) # 9

def get_fashion_mnist_labels(labels):
    """
    :param labels: 传入的数字标签
    :return: 返回标签对应的文字
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, sharex=True, sharey=True) # 创建子图
    axes = axes.flatten() # 将多维数组转换为一维数组
    for i, (ax, img) in enumerate(zip(axes, imgs)): # 将遍历的元素同时赋值给两个变量， zip()用来遍历两个列表中的元素， i 为索引，ax 为子图，img 为图片
        if th.is_tensor(img):
            # 如果是张量，则转换为 numpy，并且转换为 RGB
            ax.imshow(img.numpy())
        else:
            # 如果是 PIL 图片，则直接显示
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False) # 不显示 x 轴, ax 为子图, axes 为子图数组
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

# 进行一个小批量读取， 批量大小为 18
# for X, y in iter(data.DataLoader(mnist_train, batch_size=18)):# 获取 batch_size=18 的数据
#     show_images(X.reshape(18, 28, 28), 2, 9, get_fashion_mnist_labels(y))

"""train_iter = data.DataLoader(mnist_train,
                             batch_size=256,
                             shuffle=True,
                             num_workers=4 # 多进程读取数据
                             )"""

def load_mnist(batch_size, resize=None):
    """下载数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize: # 缩放
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # 组合转换操作
    mnist_train = thv.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = thv.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

train_iter, test_iter = load_mnist(batch_size=32, resize=64)

