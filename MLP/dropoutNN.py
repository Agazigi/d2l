import matplotlib.pyplot as plt
import torch as th
from MLPNN import load_mnist, train
from torch import nn

drop_out1, drop_out2 = 0.2, 0.5
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(drop_out1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(drop_out2),
                    nn.Linear(256, 10)
                    )

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
trainer = th.optim.SGD(net.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss(reduction='none')
num_epochs = 10
train_iter, test_iter = load_mnist(batch_size=256)
train(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()