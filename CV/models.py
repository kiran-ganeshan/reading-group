import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from tqdm import tqdm

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
    
#A = torch.Tensor([[1, 0], [0, 1], [0, 0]])
d = 100
l = 6
N = 10000
# Us = [torch.randn(d, d) for _ in range(l)]
# Ss = [torch.randn(d) for _ in range(l)]
# Vs = [torch.randn(d, d) for _ in range(l)]
torch.seed()
As = [torch.randn(d, d) for _ in range(l)]
#Bs = [torch.matrix_exp(U - U.T) @ torch.diag(S) @ torch.matrix_exp(V - V.T) for U, S, V in zip(Us, Ss, Vs)]
x = torch.randn(N, d)
def A_apply(x):
    for i, A in enumerate(As):
        x = x @ A.T
        if i != len(As) - 1:
            x = torch.tanh(x)
    return x
fig = plt.figure(figsize=(20, 15))
torch.seed()
class NNLayer(torch.nn.Module):
    def __init__(self, indim, outdim):
        super(NNLayer, self).__init__()
        self.mat = torch.nn.Parameter(torch.randn(outdim, indim))
    def forward(self, x):
        return x @ self.mat.T
class NN(torch.nn.Module):
    def __init__(self, type, indim, outdim):
        super(NN, self).__init__()
        lst = [type(indim, d)]
        lst += [type(d, d) for _ in range(l-2)]
        lst += [type(d, outdim)]
        self.lst = torch.nn.ModuleList(lst)
    def forward(self, x):
        for i, layer in enumerate(self.lst):
            x = layer(x)
            if i != len(self.lst) - 1:
                x = torch.tanh(x)
        return x 
class SVDNNLayer(torch.nn.Module):
    def __init__(self, indim, outdim):
        super(SVDNNLayer, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(outdim, outdim))
        _, S, _ = torch.svd(torch.randn(indim, outdim))
        self.s = torch.nn.Parameter(torch.log(S))
        self.v = torch.nn.Parameter(torch.randn(indim, indim))
        self.dim_diff = outdim - indim
    def forward(self, x):
        x = x @ torch.matrix_exp((self.v - self.v.T)).T
        z = torch.zeros((*x.shape[:-1], self.dim_diff))
        x = torch.concat([torch.exp(self.s) * x, z], dim=-1)
        x = x @ torch.matrix_exp((self.u - self.u.T)).T
        return x
    
lrs = [1, 0.3, 0.1, 0.03, 0.01]
n_seeds = 5
for lr in lrs:
    nn = NN(NNLayer, d, d)
    svdnn = NN(SVDNNLayer, d, d)
    opt = torch.optim.Adam(nn.parameters(), lr=lr)
    svdopt = torch.optim.Adam(svdnn.parameters(), lr=lr)
    n = 1000
    nn_losses = []
    svd_nn_losses = []
    ytarg = A_apply(x)
    for t in tqdm(range(n)):
        # if t % mult == 0:
        #     ax = fig.add_subplot(n // 4 + 1, 4, t // mult + 1, projection='3d')
        #     plot_ds_and_surface(ax, A_apply)
        #     plot_ds_and_surface(ax, lambda x: nn(x).detach())
        #     plot_ds_and_surface(ax, lambda x: svdnn(x).detach())
        
        y = nn(x)
        
        opt.zero_grad()
        loss = ((y - ytarg) ** 2).mean()
        loss.backward()
        nn_losses.append(loss.item())
        opt.step()
        
        y = svdnn(x)
        svdopt.zero_grad()
        loss = ((y - ytarg) ** 2).mean()
        loss.backward()
        svd_nn_losses.append(loss.item())
        svdopt.step()
    print(f'last nn loss: {nn_losses[-1]}')
    print(f'last svd loss: {svd_nn_losses[-1]}')
    plt.plot(list(range(len(nn_losses) - 10)), nn_losses[10:], c='tab:blue', label=f'lr={lr}')
    plt.plot(list(range(len(svd_nn_losses) - 10)), svd_nn_losses[10:], c='tab:orange', label=f'lr={lr}')
plt.legend()
plt.show()