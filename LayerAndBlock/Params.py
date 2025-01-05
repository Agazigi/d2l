import torch as th
from torch import nn

# net = nn.Sequential(nn.Linear(4, 8), # net[0]
#                     nn.ReLU(),
#                     nn.Linear(8, 1))
X = th.rand(size = (2, 4))
# print(net(X))
'''
    参数查看与访问
'''
# 通过 Sequential 创建的层可以根据 [idx] 访问参数
# print(net[0].state_dict())
# print(net[1].state_dict())
# print(net[2].state_dict())
# print(net[0].weight)
# print(net[2].bias)
# print(net[2].bias.data)
# print(*[(name, param) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# print(net.state_dict()['2.bias'].data)

# def block1():
#     return nn.Sequential(nn.Linear(4, 8),
#                          nn.ReLU(),
#                          nn.Linear(8, 4),
#                          nn.ReLU())

# def block2():
#     net = nn.Sequential()
#     for i in range(4):
#         # 在这里嵌套
#         net.add_module(f'block {i}', block1()) # 指定块的名称
#     return net
#
# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet(X))
# print(rgnet)
# print(rgnet.state_dict())

'''
    参数初始化
'''
# def init_normal(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, mean = 0, std = 0.01)
#         nn.init.zeros_(m.bias)
# net.apply(init_normal)
# print(net[0].weight.data, net[0].bias.data)
#
# def init_constant(m):
#     if type(m) == nn.Linear:
#         nn.init.constant(m.weight, 1)
#         nn.init.zeros_(m.bias)
#
# def my_init(m):
#     if type(m) == nn.Linear:
#         nn.init.uniform_(m.weight, -10, 10)
#         m.weight.data *= m.weight.data.abs() >= 5
#
# net[0].weigth.data[1] = 10


'''
    参数绑定
'''
# shared = nn.Linear(8, 8)
# net = nn.Sequential(nn.Linear(4, 8),
#                     nn.ReLU(),
#                     shared,
#                     nn.ReLU(),
#                     shared,
#                     nn.ReLU(),
#                     nn.Linear(8, 1))
# print(net.state_dict())

'''
    延时初始化
'''
# 当输入为一个未知的数时，输出的维度也是未知的，此时无法确定参数的形状，所以需要使用延时初始化。
# 就是说当第一次接受数据的时候才进行初始化。