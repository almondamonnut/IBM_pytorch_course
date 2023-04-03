# this code is to clearify my curiosity about 
# can I use these .item(), .data, and float() interchangably

import torch

a = torch.tensor(2.,requires_grad=True)

print(a)
# tensor(2., requires_grad=True)

print(a.detach()) # tensor(2.)
print(a) # tensor(2., requires_grad=True)

print(a.item())  # 2.0
print(a.data)  # tensor(2.)

print(float(a))  # 2.0
# ได้เหมือนกัน

# a.detach() = torch.tensor(4.) 
# this will cause error

# a successful change of data 
a.data = torch.tensor(4.)
print(a) # tensor(4., requires_grad=True) 