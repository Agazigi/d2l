import torch as th
from torch.nn import functional as F
import utils.Loader as ld
import math
import torch.nn as nn
from SequenceModel.myutils import Accumulator, Timer, Animator
import matplotlib.pyplot as plt

batch_size, num_steps = 32, 35
train_iter, vocab = ld.load_data_time_machine(batch_size, num_steps)

def try_gpu(i=0):
    return th.device(f'cuda:{i}') if i >= 0 and th.cuda.is_available() else th.device('cpu')

def get_params(vocab_size, num_hiddens, device):
    """
    初始化 RNN 的模型参数
    """
    num_inputs = num_outputs = vocab_size # 输入和输出都是词表的词数 28

    def normal(shape):
        """
        返回一个参数的初始化
        """
        return th.randn(size=shape, device=device) * 0.01

    # H_t = ReLU( X_t * W_xh + H_t-1 * W_hh + b_h)
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = th.zeros(num_hiddens, device=device)

    # O_t = H_t * W_hq + b_q
    W_hq = normal((num_hiddens, num_outputs))
    b_q = th.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True) # 启用梯度计算
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """
    初始化时返回隐状态
    """
    return (th.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """
    向前传播 RNN
    """
    W_xh, W_hh, b_h, W_hq, b_q = params # 获取参数
    H, = state # 获取隐藏层状态
    outputs = []
    for X in inputs: # 遍历输入
        H = th.tanh(th.mm(X, W_xh) + th.mm(H, W_hh) + b_h) # 计算隐藏层状态
        Y = th.mm(H, W_hq) + b_q # 计算输出
        outputs.append(Y) # 保存输出
    return th.cat(outputs, dim=0), (H,) # 返回输出和隐藏层状态

class RNNModelScratch: # RNN模型
    """
    RNN 模型
    """
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size= vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state= init_state
        self.forward_fn = forward_fn
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(th.float32) # 转换为one-hot向量
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device): # 初始化隐藏层状态
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params, init_rnn_state, rnn)

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

# ans = predict('time traveller ', 10, net, vocab, try_gpu())
# print(ans)

'''
    问题：对于一个长度 n 的序列，计算 n 个时间步上的梯度 -> 梯度爆炸！ 梯度消失！
    (最近读的一篇论文中也有对可能出现梯度爆炸或消失的情况进行优化。
     论文链接：)
    可以通过梯度裁剪
'''

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

'''
    训练
'''

'''
    检验
    困惑度：exp(-1/n * ( sum( log(P(x_t | X_t-1, ..., x_1)))
          下一个词元的实际选择数的调和平均数
          1: 最好
          +∞: 不好
'''
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
train_rnn(net, train_iter, vocab, lr, num_epochs, try_gpu())

