import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat("ex7data2.mat")
X = data["X"]

# K = 3
# center = X[np.random.choice(X.shape[0], K, replace=False)]


def euclidean_distance(x, center):
    return np.sqrt(np.sum((x - center) ** 2))

def kmeans(X, K, max_iters=100, tol=1e-4):
    # 选取 K 个中心
    global cluster
    centers = X[np.random.choice(X.shape[0], K, replace=False)]

    for i in range(max_iters):
        # 创建一个和数据的数量一样多的数组
        cluster = np.zeros(X.shape[0])
        for j in range(X.shape[0]): # 对每一个数据点
            # 我们计算它和每一个中心的距离
            distances = np.array([euclidean_distance(X[j], center) for center in centers])
            # 我们选取最小的一个，即这个数据点被分配给的哪一个中心
            cluster[j] = np.argmin(distances)
        # 对每一个中心，选取被分配给它的点，计算均值，更新中心
        for k in range(K):
            centers[k] = np.mean(X[cluster == k], axis=0)

    # 返回中心，和每一个点属于谁
    return centers, cluster


centers, cluster = kmeans(X, K=3)
print(centers)

# 画散点图
plt.scatter(X[:, 0], X[:, 1], c=cluster, s=50, cmap="plasma")
plt.scatter(centers[:, 0], centers[:, 1], c='r', s=200)
plt.show()