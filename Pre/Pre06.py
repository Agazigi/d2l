import torch as th

x = th.arange(4, dtype=th.float32, requires_grad=True)
print(x)
print(x.grad)

y = 2 * th.dot(x, x) # 2 * x^2
print(y)
print(y.backward())
print(x.grad) # 4 * x

x.grad.zero_() # 默认情况下，梯度会累加，需要清零
y = x.sum()
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
x = th.arange(4, dtype=th.float32, requires_grad=True)
y = th.dot(x, x)
print(y)
y.backward()
print(x.grad)