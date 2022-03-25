import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from tqdm import tqdm
import scipy.linalg
from sympy import Matrix
import wandb

class MLP(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, num_nodes, num_layers, act=torch.nn.ReLU()):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        assert num_layers > 0
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else num_nodes
            out_dim = output_dim if i == num_layers - 1 else num_nodes
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
            if i != num_layers - 1:
                self.layers.append(act)
        
    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x
        
class SVDLinearCondense(torch.nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        assert out_dim < in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.u = torch.nn.Parameter(torch.randn(in_dim, in_dim))
        self.s = torch.nn.Parameter(torch.randn(out_dim))
        self.v = torch.nn.Parameter(torch.randn(out_dim, out_dim))
        self.w = torch.nn.Parameter(torch.randn(in_dim - out_dim))
        self.bias = torch.nn.Parameter(torch.randn(out_dim))
        
    def forward(self, x):
        x = torch.matrix_exp((self.u - self.u.T) / 2) @ x
        x = x[..., :self.out_dim] * self.s
        x = torch.matrix_exp((self.v - self.v.T) / 2) @ x
        x = x + self.bias
        return x
    
    def backward(self, x):
        x = x - self.bias
        x = torch.matrix_exp((self.v.T - self.v) / 2) @ x
        x = x * self.s
        x = torch.concat([x, self.w])
        x = torch.matrix_exp((self.u.T - self.u) / 2) @ x
        return x
        
        
class SVDVAE(torch.nn.Module):
    
    def __init__(self, input_dim, latent_dim, num_nodes, num_layers, act=torch.nn.ReLU()):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        assert num_layers > 0
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else num_nodes
            out_dim = latent_dim if i == num_layers - 1 else num_nodes
            self.layers.append(SVDLinearCondense(in_dim, out_dim))
            if i != num_layers - 1:
                self.layers.append(act)
        self.std = torch.nn.Parameter(torch.randn(latent_dim))
        
    def encode(self, x):
        for mod in self.layers:
            x = mod(x)
        return x
    
    def decode(self, x):
        for mod in reversed(self.layers):
            x = mod.backward(x)
        return x

    def forward(self, x):
        mu = self.encode(x)
        x = mu + self.std * torch.randn_like(self.std)
        x = self.decode(x)
        return x, mu, self.std
        
class VAE(torch.nn.Module):
    
    def __init__(self, data_dim, latent_dim, num_nodes, num_layers, act=torch.nn.ReLU()):
        super().__init__()
        self.encoder = MLP(data_dim, latent_dim, num_nodes, num_layers, act)
        self.decoder = MLP(latent_dim, data_dim, num_nodes, num_layers, act)
        self.std = torch.randn(latent_dim)
        
    def forward(self, x):
        mu = self.encoder(x)
        x = mu + self.std * torch.randn_like(self.std)
        x = self.decoder(x)
        return x, mu, self.std

d = 50        # layer dims
l = 3         # layers
N = 10000     # training points
n = 1000      # training steps
n_seeds = 20  # seeds
svdsym = True
nnsym = True





def A_apply(As, x):
    for i, A in enumerate(As):
        x = x @ A.T
        if i != len(As) - 1:
            x = torch.tanh(x)
    return x
fig = plt.figure(figsize=(20, 15))
torch.seed()
class NNLayer(torch.nn.Module):
    def __init__(self, indim, outdim, sym):
        super(NNLayer, self).__init__()
        self.mat = torch.nn.Parameter(torch.randn(outdim, indim))
    def forward(self, x):
        return x @ self.mat.T
class NN(torch.nn.Module):
    def __init__(self, type, indim, outdim, activation=torch.tanh, sym=True):
        super(NN, self).__init__()
        lst = [type(indim, d, sym)]
        lst += [type(d, d, sym) for _ in range(l-2)]
        lst += [type(d, outdim, sym)]
        self.lst = torch.nn.ModuleList(lst)
        from torch.linalg import vector_norm as norm
        self.activation = (lambda x: activation(norm(x)) * x / norm(x)) if sym else activation
    def forward(self, x):
        for i, layer in enumerate(self.lst):
            x = layer(x)
            if i != len(self.lst) - 1:
                x = self.activation(x)
        return x 
class SVDNNLayer(torch.nn.Module):
    def __init__(self, indim, outdim, sym):
        super(SVDNNLayer, self).__init__()
        _, S, _ = torch.svd(torch.randn(indim, outdim))
        self.u = torch.nn.Parameter(6.28 * torch.randn(outdim, outdim))
        self.s = torch.nn.Parameter(S)
        if not sym:
            self.v = torch.nn.Parameter(6.28 * torch.randn(indim, indim))
        self.dim_diff = outdim - indim
        self.sym = sym
    def forward(self, x):
        if not self.sym:
            x = x @ torch.matrix_exp((self.v - self.v.T) / 2)
        z = torch.zeros((*x.shape[:-1], self.dim_diff))
        x = torch.concat([self.s * x, z], dim=-1)
        x = x @ torch.matrix_exp((self.u - self.u.T) / 2)
        return x
    
lrs = [1, 0.7, 0.3, 0.2, 0.1, 0.07, 0.03, 0.02, 0.01, 0.007, 0.003, 0.002, 0.001]

for lr in lrs:
    nn_losses = []
    svd_losses = []
    nn_test_losses = []
    svd_test_losses = []
    
    config = {'lr': lr, 'width': d, 'depth': l, 'steps': n, 'dataset_size': N, 'svdsym': svdsym, 'nnsym': nnsym}
    run = wandb.init(project='svdnn', entity='kbganeshan', reinit=True, config=config)
    
    for _ in range(n_seeds):
        
        torch.seed()
        As = [torch.randn(d, d) for _ in range(l)]
        x = torch.randn(N, d)
        ytarg = A_apply(As, x)
        nn = NN(NNLayer, d, d, sym=nnsym)
        svdnn = NN(SVDNNLayer, d, d, sym=svdsym)
        opt = torch.optim.Adam(nn.parameters(), lr=lr)
        svdopt = torch.optim.Adam(svdnn.parameters(), lr=lr)
        
        for t in tqdm(range(n)):
            
            y = nn(x)
            opt.zero_grad()
            nnloss = ((y - ytarg) ** 2).mean()
            nnloss.backward()
            opt.step()
            
            y = svdnn(x)
            svdopt.zero_grad()
            svdloss = ((y - ytarg) ** 2).mean()
            svdloss.backward()
            svdopt.step()
            
        nn_losses.append(nnloss.item())
        svd_losses.append(svdloss.item())
        
        xtest = torch.randn(N, d)
        ytarg = A_apply(As, xtest)
        y = nn(xtest)
        nn_test_losses.append(((y - ytarg) ** 2).mean())
        y = svdnn(xtest)
        svd_test_losses.append(((y - ytarg) ** 2).mean())
        
    names = ['nn_train', 'svd_train', 'nn_test', 'svd_test']
    loss_lsts = [nn_losses, svd_losses, nn_test_losses, svd_test_losses]
    for name, loss_lst in zip(names, loss_lsts):
        mean = sum(loss_lst) / n_seeds
        std = sum([(loss - mean) ** 2 for loss in loss_lst]) / n_seeds
        wandb.log({f'{name}_loss': loss_lst, 
                   f'mean_{name}_loss': mean, 
                   f'std_{name}_loss': std})
        
    run.finish()
        
    
    
plt.legend()
plt.show()
