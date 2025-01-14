import math
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from SequenceModel.myutils import Accumulator, Timer, Animator
from utils import Loader as ld
import matplotlib.pyplot as plt

batch_size = 32
num_steps = 35
train_iter, vocab = ld.load_data_time_machine(batch_size, num_steps) # 加载数据

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

state = th.zeros((1, batch_size, num_hiddens))

class RNNModel(nn.Module): # 继承父类 nn.Module
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs) # 父类 nn.Module 的初始化方法，其中的 **kwargs 是一个字典，
                                                # 包含了父类 nn.Module 的所有参数
        self.rnn = rnn_layer # rnn_layer 是 nn.RNN 的实例，即我们的 rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size # 隐藏单元数
        if not self.rnn.bidirectional: # 单向 RNN
            self.num_directions = 1 # 单向 RNN
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            # 双向 RNN
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    def forward(self, inputs, state):
        """
        向前传播
        """
        X = F.one_hot(inputs.T.long(), self.vocab_size) # 将输入转换为 one-hot 向量
        X = X.to(th.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小, 隐藏单元数)
        # 它的输出形状是(时间步数*批量大小, 词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # 当是 GRU 时
            return th.zeros((self.num_directions * self.rnn.num_layers,
                             batch_size, self.num_hiddens), device=device)
        else:
            # LSTM 一个是记忆状态，另一个是隐藏状态
            return (th.zeros((self.num_directions * self.rnn.num_layers,
                             batch_size, self.num_hiddens), device=device),
                    th.zeros((self.num_directions * self.rnn.num_layers,
                             batch_size, self.num_hiddens), device=device))

device = th.device(f'cuda:{0}') if th.cuda.is_available() else th.device('cpu')
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net.to(device)


'''
    重复的内容
'''
def predict_rnn(prefix, num_preds, net, vocab, device):
    """
    预测
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: th.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """
    梯度裁剪
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = th.sqrt(sum(th.sum((p.grad ** 2)) for p in params).clone().detach()) # 求得每一个参数的 L2 范数
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm # 进行裁剪

def train_epoch_rnn(net, train_iter, loss, optimizer, device, use_random_iter):
    state = None # 初始化 state
    metric = Accumulator(2)
    timer = Timer()
    for X, Y in train_iter: # 遍历数据集
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device) # 在第一次迭代或者是随机抽样时初始化 state
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_() # GRU
            else:
                for s in state:
                    s.detach_() # LSTM
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device) # 转换为 device
        y_hat, state = net(X, state) # 前向传播
        l = loss(y_hat, y.long()).mean() # 计算损失
        if isinstance(optimizer, th.optim.Optimizer):
            optimizer.zero_grad() # 清空梯度
            l.backward() # 反向传播
            grad_clipping(net, 1) # 裁剪梯度
            optimizer.step() # 更新参数
        else:
            l.backward() # 反向传播
            grad_clipping(net, 1) # 裁剪梯度
            optimizer(batch_size=1) # 更新参数
        metric.add(l * y.numel(), y.numel()) # 更新损失和词数
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop() # 返回困惑度和词数

def train_rnn(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    global ppl, speed # 困惑度和速度
    loss = nn.CrossEntropyLoss() # 交叉熵损失
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        optimizer = th.optim.SGD(net.parameters(), lr=lr)
    else:
        def sgd(params, lr, batch_size):
            with th.no_grad():
                for param in params:
                    param -= lr * param.grad / batch_size
                    param.grad.zero_()

        optimizer = lambda batch_size: sgd(net.params, lr, batch_size) # 优化函数
    predict = lambda prefix: predict_rnn(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_rnn(net, train_iter, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, ', f'{speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    plt.show()

num_epochs, lr = 500, 1
train_rnn(net, train_iter, vocab, lr, num_epochs, device)