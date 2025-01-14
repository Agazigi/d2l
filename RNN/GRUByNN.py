import torch as th
import torch.nn as nn
import utils.Loader as lm
from RNN.RNN import train_rnn
from RNN.RNNByNN import RNNModel

batch_size = 32
num_steps = 35
train_iter, vocab = lm.load_data_time_machine(batch_size=batch_size, num_steps=num_steps)

num_inputs = len(vocab) # 输入维度为词表大小\
num_hiddens = 256
GRU_Layer = nn.GRU(num_inputs, num_hiddens) # 创建 GRU 层
net = RNNModel(GRU_Layer, vocab_size=len(vocab))
device = th.device(f'cuda:{0}') if th.cuda.is_available() else th.device('cpu')
net.to(device)
train_rnn(net, train_iter, vocab, lr=1, num_epochs=500, device=device)