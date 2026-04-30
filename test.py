import torch

x1 = torch.tensor([[0., 0.],
                   [1., 0.]])

x2 = torch.tensor([[0., 1.],
                   [1., 1.]])

dist = torch.cdist(x1, x2)
print(dist)