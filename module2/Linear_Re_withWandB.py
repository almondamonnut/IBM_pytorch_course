# This code demonstrate how to train Linear Regression with
# both weights and biases being used to train and update

import torch
import matplotlib.pyplot as plt

def forward(x):
    return w*x+b

def losskung(yhat,y):
    return torch.mean((yhat-y)**2)

# define parameters
w = torch.tensor(-10., requires_grad=True)
b = torch.tensor(-15., requires_grad=True)

# define x, f, and y
x = torch.arange(-3, 3, 0.1).view(-1, 1)
f = x - 1
y = f + 0.1 * torch.randn(f.size())
plt.plot(x, y, '.')
plt.show()

lr = 0.1

# training loop
for epoch in range(25):
    yhat = forward(x)
    loss = losskung(yhat, y)
    loss.backward()

    # update parameters
    w.data = w.data - lr*w.grad.data
    w.grad.zero_()

    b.data = b.data - lr*b.grad.data
    b.grad.zero_()

    plt.plot(x, y.data, 'b.')
    plt.plot(x, yhat.data, 'r')
    plt.show()
    print(f"epoch{epoch}: loss =", loss)