import torch
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def forward(x):
    return w*x + b

def losskung(yhat, y):
    return torch.mean((yhat-y)**2)

w = torch.tensor(-10., requires_grad=True)
b = torch.tensor(-12., requires_grad=True)

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3*X
Y = f + torch.randn(f.size()) * 0.1

lr = 0.01

# training loop
for epoch in range(5):
    for x,y in zip(X,Y):
        # clear the previous plot
        plt.cla()

        yhat = forward(x)
        ax.plot(X, Y.data, 'b.')
        # print(forward(X).shape)
        ax.plot(X, forward(X).data, 'r')
                
        ax.set_xlim([-3.5, 3.5]) 
        ax.set_ylim([-10, 10])
        
        # reload
        plt.draw()
        plt.pause(0.05)

        loss = losskung(yhat, y)
        loss.backward()
        print(f"epoch#{epoch}: loss={loss}")
        
        # update parameters
        w.data = w.data - lr*w.grad.data
        w.grad.zero_()
        
        b.data = b.data - lr*b.grad.data
        b.grad.zero_()

plt.show()