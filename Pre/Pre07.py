import torch as th
from torch.distributions import multinomial
import matplotlib.pyplot as plt

fair_probs = th.ones(6) / 6 # 定义6个等概率的离散随机变量
counts = multinomial.Multinomial(10, fair_probs).sample(th.Size([200])) # 模拟500个10次掷骰子
print(counts)
counts = counts.cumsum(dim = 0)
print(counts)
estimate = counts / counts.sum(dim = 1, keepdim = True)
print(estimate)

plt.figure()
for i in range(6):
    plt.plot(estimate[:, i].numpy(), label=f'Point {i+1}')
plt.xlabel('Simulation Rounds')
plt.ylabel('Estimated Probability')
plt.title('Estimated Probabilities of Dice Points Over Time')
plt.legend()
plt.show()