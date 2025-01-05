import torch as th
from torch import nn
from torch.nn import functional as F

# x = th.arange(4)
# th.save(x, 'x-file')

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

'''
    save 和 load
'''

#  保存架构
# net = MLP()
# X = th.rand(size=(2,20))
# Y = net(X)
# th.save(net.state_dict(), 'mlp.params') # 保存参数

# clone = MLP()
# clone.load_state_dict(th.load('mlp.params')) # 加载参数
# print(clone.eval()) # 评估模式

