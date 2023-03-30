# This code is to demonstrate calculation of gradient in pytorch

import torch

# initialize a tensor with parameter requires_grad = True 
# to secify that we will calculate gradient of this tensor.
# note that IntTensor cannot require gradient
# x = torch.tensor(2, requires_grad=True) RuntimeError: Only Tensors of floating point and complex dtype can require gradients

x = torch.tensor(2., requires_grad=True)
y = x**2  # create a tensor y equaling to x squared

# calculate the derivative of y using tensor.backward()
y.backward()   # return None, just calculating inside

# plug the value x=2 in the derivative of y
print(x.grad)   # tensor(4.)
print(x.grad)   # tensor(4.)
# x.grad is just a value stored, not a method of changing gradient

# y.backward()   If you backward the second time, it wil raise error
# print(x.grad)

# If you want to accumulate the gradient, define the tensor y again before calculating grad
y = x**2  
y.backward()
print(x.grad)   # tensor(8.)


#______________________________________


# assinging grad and deleting grad

# initialize with requires_grad
x = torch.tensor(2., requires_grad=True)   
print(x)    # tensor(2., requires_grad=True)

# method 1 set requires_grad to false
x.requires_grad_(False) 
print(x)    # tensor(2.)
# it is now just ta normal tensor

# method 2 .detach()
print(x.detach())   # tensor(2.)
x = x.detach()  
print(x)  # tensor(2.)
 

# accumulation of gradient
x.requires_grad_(True)

for epoch in range(3):
    y = 3*x**2
    y.backward() 

    print(x.grad)

# You'll see the gradient is added up
# tensor(12.)
# tensor(24.)
# tensor(36.)

# This is how we reset the gradient accumulation
for epoch in range(3):

    x.grad.zero_()
    y = 3*x**2
    y.backward() 
    print(x.grad)

# now the gradient is reseted every new epoch
# tensor(12.)
# tensor(12.)
# tensor(12.)

#______________________________________

# partial derivatives
u = torch.tensor(1., requires_grad=True)
v = torch.tensor(2., requires_grad=True)
f = u * v
f.backward()
print(u.grad)
print(v.grad)
