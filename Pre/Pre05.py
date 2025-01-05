import torch as th
from torch.onnx.symbolic_opset9 import linalg_norm

'''
    线性代数
'''

# x = th.tensor(3)
# y = th.tensor(2)
# print(x + y)
#
# z = th.tensor([1, 2, 3, 4, 5])
# print(z)
# print(z[3])
# print(len(z))
# print(z.shape)

# a = th.arange(20, dtype=th.float64).reshape(4,5)
# print(a)
# print(a.T)
# print(a[2,2])
# print(a[:,2])
#
# b = th.ones(3,3,dtype=th.float64)
# print(b==b.T)

# a = th.arange(24, dtype=th.float64).reshape(2,3,4)
# print(len(a))
# print(a.shape)
# b = a.clone() # 通过重新分配内存，对a进行修改不会影响b
# print(a)
# b[1,1,1] = 100
# print(b)F
# print(a)
# print(a+b)

# a = th.arange(24, dtype=th.float64).reshape(4,6)
# print(a)
# print(a)
# b = a.sum(dim=1, keepdim=True)
# print(b)
# c = a.sum(dim=[0,1])
# print(c)
#
# print(a/b)
# print(a.cumsum(dim=0)) # 累加

# x = th.ones(4,dtype=th.float64)
# y = th.arange(4,dtype=th.float64)
# x = x / x.sum()
# y = y / y.sum()
# print(x)
# print(y)
# print(th.dot(x,y))

# A = th.arange(16, dtype=th.float64).reshape(4,4)
# print(A)
# x = th.arange(4, dtype=th.float64)
# print(x)
# print(th.mv(A,x))

# A = th.arange(16, dtype=th.float64).reshape(4,4)
# print(A)
# B = th.arange(16, dtype=th.float64).reshape(4,4)
# print(B)
# print(th.mm(A,B))

# v = th.tensor([1,2,3], dtype=th.float64)
# print(th.norm(v)) # L_2 范数
# print(th.norm(v, p=1)) # L_1 范数
# print(th.norm(v, p=float('inf'))) # L_inf 范数
# A = th.ones(4,4)
# print(A)
# print(th.norm(A, p='fro')) # L_2 范数 Frobenus 范数


A = th.arange(24, dtype=th.float64).reshape(2,3,4)
print(A)
print(A.sum(dim=0))
print(A.sum(dim=1))
print(A.sum(dim=2))