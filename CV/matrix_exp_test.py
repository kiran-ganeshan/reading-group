from time import time
from math import log2
from tqdm import tqdm
import torch

x, y = [], []
for n in tqdm([int(2 ** (12 + k / 5)) for k in range(10)]):
    x.append(log2(n))
    X = torch.randn(n, n)
    start = time()
    A = torch.matrix_exp(X)
    end = time() - start
    y.append(log2(end))
    
print(x, y)