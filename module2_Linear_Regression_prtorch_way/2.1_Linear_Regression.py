# this code is to demonstrate how to write linear regression in pytorch
# equation is y = b + wx

import torch

# version 1: without torch.nn

w = torch.tensor(2., requires_grad=True)
b = torch.tensor(-1.,requires_grad=True)

# create a forward function (in torch, forward = predict)
def forward(x):
    return b + w*x

# let's test some values of x
x = torch.tensor(1.)
print(forward(x))
# tensor(1., grad_fn=<AddBackward0>)

x = torch.tensor([1.])
print(forward(x))
# tensor([1.], grad_fn=<AddBackward0>)

x = torch.tensor([1., 2.])
print(forward(x))
# tensor([1., 3.], grad_fn=<AddBackward0>)

x = torch.tensor([[1.],[4.]])
print(forward(x))
# tensor([[1.],
#         [7.]], grad_fn=<AddBackward0>)

x = torch.tensor([[1.,2.],[3.,6.]])
print(forward(x))
# tensor([[ 1.,  3.],
#         [ 5., 11.]], grad_fn=<AddBackward0>)

# conclude: torch tensors can strech themselves to add or multiply
# any shape of other tensors


#________________________________________


# version 2: with torch.nn

from torch.nn import Linear

torch.manual_seed(1)

# สร้าง model ที่เป็น instance ของ torch.nn.Linear
# in_features = 1 แปลว่า แต่ละ row ของ x ต้องเป็น tensor 1d
model = Linear(in_features=1, out_features=1)

# ส่อง parameters เริ่มต้นของ model
print(list(model.parameters()))
# [Parameter containing:
# tensor([[0.5153]], requires_grad=True), Parameter containing:
# tensor([-0.4414], requires_grad=True)]
# เลขพวกนี้เป็นเลขของการ random init ตาม seed ที่เราเซ็ตไว้

# let's test some values of x
# x = torch.tensor(1.)
# print(model(x))
# RuntimeError: both arguments to matmul need to be at least 1D, but they are 0D and 2D

x = torch.tensor([1.])
print(model(x))
# tensor([0.0739], grad_fn=<AddBackward0>)

# x = torch.tensor([1., 2.])
# print(model(x))
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x2 and 1x1)

x = torch.tensor([[1.], [243.], [22.]])
print(model(x))
# tensor([[7.3885e-02],
#         [1.2477e+02],
#         [1.0894e+01]], grad_fn=<AddmmBackward0>)


#________________________________________

# version 3: build a custom module

from torch import nn
# import torch.nn as nn

class LR(nn.Module):
    def __init__(self, in_size, output_size):

        # calling super constructor to create objects in torch.nn.Module
        super(LR, self).__init__()

        # This cannot be called without calling super constructor first
        self.linear = nn.Linear(in_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LR(1,1)
print(model(torch.tensor([1.])))


# checking parameters with their names in model

print(model.state_dict())
# OrderedDict([('linear.weight', tensor([[-0.1939]])), ('linear.bias', tensor([0.4694]))])

# call a parameter from parameters in state_dict()
print(model.state_dict()['linear.weight'])  # tensor([[-0.1939]])

# change the value o fthat parameter in state_dict()
# model.state_dict()['linear.weight'] = -0.3   แบบนี้ค่าไม่เปลี่ยน
# วิธีที่ 1
model.state_dict()['linear.weight'][0] = -0.3
print(model.state_dict())
# OrderedDict([('linear.weight', tensor([[-0.3000]])), ('linear.bias', tensor([0.4694]))])

# วิธีที่ 2
print(model.state_dict()['linear.weight'].data) # tensor([[-0.1939]])
model.state_dict()['linear.weight'].data[0] = -0.2
print(model.state_dict())
# OrderedDict([('linear.weight', tensor([[-0.2000]])), ('linear.bias', tensor([0.4694]))])


# printing only keys/values of the state_dict()
print(model.state_dict().keys()) 
# odict_keys(['linear.weight', 'linear.bias'])
print(model.state_dict().values())
# odict_values([tensor([[-0.2000]]), tensor([0.4694])])