import torch

def conjugate_gradient(Hv, v, iters, tol=1e-10):
    p = v.clone()
    r = v.clone()
    x = torch.zeros_like(v)
    inner = torch.dot(r, r)
    
    for _ in range(iters):
        z = Hv(p)
        u = inner / torch.dot(p, z)
        x += u * p
        r -= u * z
        new_inner = torch.dot(r, r)
        p = r + new_inner / inner * p
        inner = new_inner
        if inner < tol:
            break
        
    return x

