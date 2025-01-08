import torch as th
from torch.nn import functional as F
import utils.Loader as ld
import math
import torch.nn as nn

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

def predict(prefix, num_preds, net, vocab, device):
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

ans = predict('time traveller ', 10, net, vocab, try_gpu())
print(ans)