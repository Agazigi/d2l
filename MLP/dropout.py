import matplotlib.pyplot as plt
import torch as th
from torch import nn
from MLPNN import train,load_mnist

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return th.zeros_like(X)
    if dropout == 0:
        return X
    mask = (th.rand(X.shape) > dropout).float()
    return mask * X / (1 - dropout)

drop_out = 0.5

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        if self.training == True:
            H1 = dropout_layer(H1, drop_out)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, drop_out)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = load_mnist(batch_size)
trainer = th.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()