import torch as th

'''
    创建张量
'''

a = th.arange(12, dtype=th.int32)
print(f'张量 a 的值为: {a}')
b = a.view(3,4)
print(f'矩阵 b 的值为: {b}')
print(f'矩阵 b 的维度为: {b.dim()}') # 查看矩阵的维度
print(f'矩阵 b 的大小为: {b.size()}')
print(f'矩阵 b 的形状为: {b.shape}')

print(f'矩阵 b 中元素的个数为: {b.numel()}') # 查看矩阵中元素的个数
print(f'矩阵 b 的形状为: {b.size(0)} , {b.size(1)}')

c = a.reshape(4, 3)
print(f'矩阵 c 的值为: {c}')

d = th.zeros(4, 4, dtype=th.int32) # 创建一个全零矩阵
print(f'矩阵 d 的值为: {d}')

e = th.ones(4, 4, dtype=th.int32) # 创建一个全1矩阵
print(f'矩阵 e 的值为: {e}')

f = th.randn(4, 4) # 创建一个随机矩阵
print(f'矩阵 f 的值为: {f}')

g = th.tensor([[1,2,3], # 利用列表创建矩阵
                [4,5,6],
                [7,8,9]])
print(f'矩阵 g 的值为: {g}')