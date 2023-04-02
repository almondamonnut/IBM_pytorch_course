import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# create a Dataset subclass
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        f = -3*self.x
        self.y = torch.randn(f.size())*0.1 + f
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

dataset = Data()

# create a DataLoader
trainloader = DataLoader(dataset=dataset, batch_size=1)

# create a custom module
class LR(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(LR,self).__init__()
        self.linear = nn.Linear(in_shape, out_shape)
    
    def forward(self, x):
        return self.linear(x)

model = LR(1, 1)

# use a pre-defined loss function
losskung = nn.MSELoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
# check all states inside optimizer
print(optimizer.state_dict())

# for visualization purpose
X = dataset.x
Y = dataset.y

# training loop
for epoch in range(2):
    for x, y in trainloader:
        plt.cla()
        yhat = model(x)

        ax.plot(X, Y.data, 'b.')
        # print(model(X))
        ax.plot(X, model(X).data, 'r')

        plt.draw()
        plt.pause(0.05)

        loss = losskung(yhat, y)

        # reset gradients before calculating loss
        optimizer.zero_grad()

        loss.backward()

        # update all learnable parameters of the model at once
        optimizer.step()

        print(f"epoch#{epoch}: loss = {loss}")

plt.show()