import torch as th
import torch.nn as nn
import utils.Loader as ld
from RNN.RNN import train_rnn

batch_size = 32
num_steps = 35
train_iter, vocab = ld.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return th.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                th.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three() # 输入门参数
    W_xf, W_hf, b_f = three() # 遗忘门参数
    W_xo, W_ho, b_o = three() # 输出门参数
    W_xc, W_hc, b_c = three() # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = th.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    """
    初始化时返回隐状态
    """
    return (th.zeros((batch_size, num_hiddens), device=device),
            th.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    """
    LSTM 向前传播
    """
    (H, C) = state
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    outputs = []
    for X in inputs:
        I = th.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = th.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = th.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = th.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * th.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return th.cat(outputs, dim=0), (H, C)

vocab_size = len(vocab)
num_hiddens = 256
device = th.device('cuda:0') if th.cuda.is_available() else th.device('cpu')
num_epochs = 500
lr = 1
train_rnn(lstm, train_iter, vocab, lr, num_epochs, device)