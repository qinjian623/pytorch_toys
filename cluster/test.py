import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.cluster.vq import whiten
from cluster.kmeans import kmeans


if __name__ == '__main__':
    pts = 25000
    a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts * 100)  # Unbalanced dataset
    b = np.random.multivariate_normal([30, 10], [[10, 2], [2, 1]], size=pts)
    c = np.random.multivariate_normal([60, 20], [[10, 2], [2, 1]], size=pts)
    d = np.random.multivariate_normal([20, 30], [[10, 2], [2, 1]], size=pts)
    features = np.concatenate((a, b, c, d))
    # Whiten data
    whitened = whiten(features)
    # Half type to save more GPU memory.
    pt_whitened = torch.from_numpy(whitened).half().cuda()

    # kmeans
    torch.cuda.synchronize()
    s = time.time()
    codebook, distortion = kmeans(pt_whitened, 100, batch_size=6400000, iter=1)
    torch.cuda.synchronize()
    e = time.time()
    print("Time: ", e-s)

    # Show
    plt.scatter(whitened[:, 0], whitened[:, 1])
    plt.scatter(codebook.cpu()[:, 0], codebook.cpu()[:, 1], c='r')
    plt.show()
