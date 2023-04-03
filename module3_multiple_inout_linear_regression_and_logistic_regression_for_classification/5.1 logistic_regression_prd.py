# This code is to demonstrate logistic regression prediction in pytorch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ploting sigmoid function that take z (seperation line between two groups) as input
z = torch.arange(-10, 10, 0.1).view(-1,1)
# way1
sig = nn.Sigmoid()
yhat = sig(z)
plt.plot(z, yhat)
plt.show()  # ได้กราฟรูปตัว S สวย ๆ

# way2 (easier)
yhat = torch.sigmoid(z)
plt.plot(z, yhat)
plt.show()
# ได้เหมือนกันเป๊ะ