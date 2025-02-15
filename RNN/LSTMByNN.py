import torch as th
import torch.nn as nn
import utils.Loader as ld
from RNN.RNN import train_rnn
from RNN.RNNByNN import RNNModel

batch_size = 32
num_steps = 35
train_iter, vocab = ld.load_data_time_machine(batch_size, num_steps)

num_inputs = len(vocab)
num_hiddens = 256
LSTM_Layer = nn.LSTM(num_inputs, num_hiddens)

net = RNNModel(LSTM_Layer, vocab_size=len(vocab))

num_epochs = 500
lr = 1
device = th.device('cuda:0') if th.cuda.is_available() else th.device('cpu')
net = net.to(device)
train_rnn(net, train_iter, vocab, lr, num_epochs, device)