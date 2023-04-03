import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

class Data(Dataset):
    def __init__(self, train = True):
        self.x = torch.arange(-1, 3, 0.1).view(-1, 1).double()
        f = (self.x * -3 + 1)
        self.y = (f + torch.rand(f.size()) * 0.4)
        self.len = self.x.size()[0]

        # create some outliers
        if train:
            self.y[0] = 0
            self.y[50:55] = 20

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
train_data = Data()
val_data = Data(train=False)

class LinearRegression(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_size, out_size, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(1, 1)

losskung = nn.MSELoss()

trainloader = DataLoader(dataset=train_data, batch_size=1)

epochs = 10
# epochs = 1
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# learning_rates = [0.01, 0.1]
val_error = torch.zeros(len(learning_rates))
train_error = torch.zeros(len(learning_rates))
MODELS = []

for i, lr in enumerate(learning_rates):
    model = LinearRegression(1,1)
    optimizer = optim.SGD(model.parameters(), lr = lr)

    for epoch in range(epochs):
        for x, y in trainloader:
            # # remove the previous line
            # plt.cla()
            # # plot train data
            # ax.plot(train_data.x, train_data.y.data, 'b.')
            # # plot prediction
            # ax.plot(train_data.x, model(train_data.x).data, 'r')

            # # วาด animation
            # plt.draw()
            # plt.pause(0.000001)

            yhat = model(x)
            loss = losskung(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)
    
    yhat = model(train_data.x)
    loss = losskung(yhat, train_data.y)
    train_error[i] = loss.item()
    MODELS.append(model)

    yhat = model(val_data.x)
    loss = losskung(yhat, val_data.y)
    val_error[i] = loss.item()

print("learning rates:", learning_rates)
print("train_error:", train_error)
print("val_error:",val_error)

# plt.show()

#too high learning rate results in explosion

# visualize LR vs ERRORS
plt.semilogx(learning_rates, train_error.numpy(), 'r')
plt.semilogx(learning_rates, val_error.numpy(), 'b')
plt.xticks(np.array(learning_rates))
plt.show()

# visualize all prediction lines
plt.plot()

# select the best model
val_error = list(val_error)
print(val_error.index(min(val_error)))
model = MODELS[val_error.index(min(val_error))]

