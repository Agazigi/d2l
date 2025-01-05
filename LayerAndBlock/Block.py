import torch as th
from torch import nn
from torch.nn import functional as F

# net = nn.Sequential(nn.Linear(20, 256),
#                     nn.ReLU(),
#                     nn.Linear(256, 10))
# # Sequential 是用来构建网络，表示一个块的类，维护了一个 ModuleList，
# X = th.rand(2, 20)
# print(net(X))


'''
    块
'''
# class MLP(nn.Module): # MLP 继承 nn.Module，是 nn.Module 的子类
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#     def forward(self, X):
#         return self.out(F.relu(self.hidden(X)))
#
# X = th.rand(2, 20)
# net = MLP()
# print(net(X))


'''
    顺序块
'''
# class MySequential(nn.Module):
#     def __init__(self, *args): # args 是一个可迭代对象，比如列表，元组等
#         super().__init__()
#         for idx, module in enumerate(args):
#             self._modules[str(idx)] = module
#     def forward(self, X):
#         for block in self._modules.values(): # _modules 是一个 OrderedDict
#             X = block(X) # 顺序执行
#         return X
#
# net = MySequential(nn.Linear(20, 256),
#                     nn.ReLU(),
#                     nn.Linear(256, 10))
# X = th.rand(2, 20)
# print(net(X))


'''
    
'''
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = th.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    def forward(self, X):
        X = self.linear(X) # 线性层
        X = F.relu(th.mm(X, self.rand_weight) + 1) # 矩阵乘法
        X = self.linear(X) # 线性层
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    def forward(self, X):
        return self.linear(self.net(X))
X = th.rand(2, 20)
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
