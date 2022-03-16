import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from tqdm import tqdm
N = 10
d = 1000
all_A_s = []
all_S_s = []
for _ in tqdm(range(N)):
    torch.seed()
    A = torch.randn(d, d)
    U = torch.randn(d, d)
    _, S, _ = torch.svd(torch.randn(d, d))
    V = torch.randn(d, d)
    u, s, v = torch.svd(A)
    all_A_s.extend([torch.log(v).item() for v in s])
    all_S_s.extend([torch.log(v).item() for v in S])
print(len(all_A_s), len(all_S_s))
bins = np.linspace(-3 * int(np.sqrt(d)), 3 * int(np.sqrt(d)), 3 * int(np.sqrt(d)))
plt.hist(all_A_s, bins, alpha=0.5, label='Natural')
plt.hist(all_S_s, bins, alpha=0.5, label='SVD')
plt.show()
    
    
A = torch.randn(3, 2)
x = torch.randn(10000, 2)
#fig = plt.figure(figsize=(20, 15))
torch.seed()


def plot_ds_and_surface(ax, apply):
    y = apply(x).numpy()
    p = np.linspace(-3, 3, 10)
    hx, hy = np.meshgrid(p, p)
    h = apply(torch.Tensor(np.stack((hx, hy), axis=-1))).numpy()
    ax.scatter(y[:, 0], y[:, 1], y[:, 2])
    #ax.plot_surface(h[..., 0], h[..., 1], h[..., 2], alpha=0.5)