import torch as th
import utils.Loader as ld
from RNN.RNN import RNNModelScratch, try_gpu, train_rnn

batch_size = 32 # 批量大小
num_steps = 35 # 时间步
train_iter, vocab = ld.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    """
        返回 GRU 的参数
    """
    num_input = num_outputs = vocab_size

    def normal(shape):
        return th.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_input, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                th.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three() # 更新门参数
    W_xr, W_hr, b_r = three() # 重置门参数
    W_xh, W_hh, b_h = three() # 候选隐藏层参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = th.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    """
        初始化隐藏层状态
    """
    return (th.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    """
        GRU 向前传播
    """
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = th.sigmoid((X @ W_xz) + (H @ W_hz) + b_z) # 更新门
        R = th.sigmoid((X @ W_xr) + (H @ W_hr) + b_r) # 重置门
        H_tilde = th.tanh((X @ W_xh) + (R * H @ W_hh) + b_h) # 候选隐藏层
        H = Z * H + (1 - Z) * H_tilde # 更新隐藏层状态
        Y = H @ W_hq + b_q # 计算输出
        outputs.append(Y)
    return th.cat(outputs, dim=0), (H,)

num_hiddens = 256
vocab_size = len(vocab)
device = try_gpu()
num_epochs = 500
lr = 1
net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
train_rnn(net, train_iter, vocab, lr, num_epochs, device)