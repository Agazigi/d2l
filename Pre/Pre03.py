import torch as th

'''
    广播机制
'''

a = th.arange(3, dtype=th.int32).reshape(3,1)
b = th.arange(2, dtype=th.int32).reshape(1,2)
print(f'a = \n{a}')
print(f'b = \n{b}')
print(f'a + b = \n{a + b}')

# 内存节省
c = th.arange(6, dtype=th.int32).reshape(2,3)
d = th.zeros_like(c, dtype=th.int32)
print(f'id(d) = {id(d)}')
d[:] = d + c # 或者是 d += c
print(f'id(d) = {id(d)}')
print(f'd = \n{d}')


# 转化为其他对象
A = a.numpy()
print(f'A = \n{A}')
print(f'type(A) = {type(A)}')
B = th.tensor(A)
print(f'B = \n{B}')
print(f'type(B) = {type(B)}')

f = th.tensor([1], dtype=th.float32)
C = f.item()
print(f'C = {C}')
print(f'type(C) = {type(C)}')