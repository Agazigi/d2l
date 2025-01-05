import torch as th
import matplotlib.pyplot as plt

# ReLU 函数
# x = th.arange(-8.0, 8, 0.1, requires_grad=True)
# y = th.relu(x)
# plt.plot(x.detach(), y.detach())
# plt.xlabel('x')
# plt.ylabel('relu(x)')
# plt.grid(True)

# fig, ax = plt.subplots(1,1, figsize=(5, 3))
# ax.plot(x.detach(), y.detach())
# ax.grid(True)
# ax.set_xlabel('x')
# ax.set_ylabel('relu(x)')
#
# y.backward(gradient=th.ones_like(y), retain_graph=True)
# # 其中 th.ones_like(y) 是一个全 1 的向量，用来表示对 y 的梯度，
# # retain_graph=True 表示在计算完 y 的梯度之后，保留计算图，以便后面计算 x 的梯度。
# plt.plot(x.detach(), x.grad.detach()) # 绘制 x 的梯度


alpha = 0.1
x = th.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = th.max(th.zeros_like(x), x) + alpha * th.min(th.zeros_like(x), x)
plt.xlabel("x")
plt.ylabel("fixed_relu(x)")
plt.grid(True)
plt.plot(x.detach(), y.detach())
plt.show()

# x = th.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = th.sigmoid(x)
# plt.plot(x.detach(), y.detach())
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('sigmoid(x)')
# plt.show()