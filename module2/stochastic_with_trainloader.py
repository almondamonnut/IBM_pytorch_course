import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

fig, ax = plt.subplots()

def forward(x):
    return w*x + b

def losskung(yhat, y):
    return torch.mean((yhat-y)**2)

w = torch.tensor(-10., requires_grad=True)
b = torch.tensor(-12., requires_grad=True)

# build a Dataset subclass
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1,1)
        f = -3*self.x
        self.y = 0.1*torch.rand(f.size()) + f
        self.len = self.x.size()[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
dataset = Data()
trainloader = DataLoader(dataset=dataset, batch_size=1)
X = dataset.x
Y = dataset.y

lr = 0.1

# training loop
for epoch in range(10):
    for x,y in trainloader:
        # clear the previous plot
        plt.cla()

        yhat = forward(x)
        ax.plot(X, Y.data, 'b.')
        ax.plot(X, (w*X+b).data, 'r')
        
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