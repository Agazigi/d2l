import torch as th

'''
    运算符
'''

# 按元素运算
a = th.tensor([1,2,3])
b = th.tensor([4,5,6])
print(f'运算 +: {a + b}')
print(f'运算 -: {a - b}')
print(f'运算 *: {a * b}')
print(f'运算 /: {a / b}')
print(f'运算 **: {a ** b}')
print(f'运算 %: {a % b}')
print(f'运算 //: {a // b}')
print(f'运算 &: {a & b}')
print(f'运算 |: {a | b}')
print(f'运算 ^: {a ^ b}')
print(f'运算 <<: {a << b}')
print(f'运算 exp: {th.exp(a)}')

c = th.arange(12, dtype=th.float32).reshape(3,4)
d = th.tensor([[1,2,2,4],[5,6,7,8],[9,10,11,12]])
e = th.cat((c,d), dim=1)
print(f'cat: {e}')
f = th.cat((c,d), dim=0)
print(f'cat: {f}')

print(f'C == D ? : {c == d}')

print(c)
print(c.sum(dim=1))