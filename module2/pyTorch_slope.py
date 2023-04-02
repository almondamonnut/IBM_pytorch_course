import torch
import matplotlib.pyplot as plt

# 1. The Hard way

w = torch.tensor(-1., requires_grad=True)
X = torch.arange(-1, 3, 0.1).view(-1, 1) # view as only 1 column
f = -3*X

plt.plot(X, f)
plt.show()