# This code is demonstration of many useful codes
# to use with torch tensor 

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# section 1 : initialization and type

# initialize a normal 1d torch tensor
a = torch.tensor([7,8,9,20])

# print type of data stored in the tensor
print(a.dtype)   # torch.int64

# print type of tensor
print(a.type())  # torch.LongTensor

# initialize tensor with specified type
a = torch.tensor([2.,3.,4.6,3.2], dtype=torch.int32) 
print(a, a.dtype) # torch.int32 & float->int ปัดลงทุกกรณี

# initialize tensor with specified type 2
a = torch.FloatTensor([1.,3])
print(a.dtype)  # torch.float32

a = torch.IntTensor([2,3])
print(a.dtype) # torch.int32

# type casting
a = a.type(torch.FloatTensor)
print(a.type(), a.dtype)

b = a.type(torch.int32)  # ข้างในวงเล็บจะใส่เป็น dtype หรือ type ของ tensor ก็ได้
print(b.type(), b.dtype)


# _____________________________________

# section 2 : some common properties of a tensor

# check size
print(b.size())  # torch.Size([2])

# check dimension
print(b.ndimension()) # 1

# dimension casting
a = a.view(2,1)   # the 1st argument = number of rows,  the 2nd argument = number of columns
print(a, a.ndimension()) # 2 dimension of 1 column

# dimension casting, auto cast the number of column to rows
print(b)
b = b.view(-1, 1)  # you can use -1 to say "the number of columns"
print(b, b.ndimension())


# _____________________________________


# section 4 : casting between tensor and the others


# numpy to torch
np_arrayA = np.array([1,2])
tensorA = torch.from_numpy(np_arrayA)
print(np_arrayA, type(np_arrayA))
print(tensorA, type(tensorA))

# convert torch to numpy array
np_arrayB = tensorA.numpy()
print(np_arrayB, type(np_arrayB))

# note that using torch.from_numpy() or .numpy() are just pointing the variable, 
# if one variable change, the other will also change the value it stores

# pandas to tensor
seriesS = pd.Series([1,2,3,4,5,6])
# 1. use pandas's function named ".values" to convert pd.series -> numpy array
print(type(seriesS.values)) # <class 'numpy.ndarray'>
# 2. now we can use torch.from_numpy() as usual
tensorS = torch.from_numpy(seriesS.values)

# tensor to list
listT = tensorS.tolist()
print(listT, type(listT)) # [1, 2] <class 'list'>


# _____________________________________


# section 5 : assigning values to tensor items

# get value of each tensor element
print(tensorA[0])  # tensor(1, dtype=torch.int32) this doesn't give us the primitive type of value in python
print(tensorA[0].item(), type(tensorA[0].item()))  # 1 และเป็น <class 'int'>

# change value of an element of tensor
tensorA[0] = 100
print(tensorA) # tensor([100, 2])

# Changing values of multiple indexes

practice_tensor = torch.tensor([2, 7, 3, 4, 6, 2, 3, 1, 2])
# just input as a list in the []
practice_tensor[[3,4,7]] = 0
print(practice_tensor)  # tensor([2, 7, 3, 0, 0, 2, 3, 0, 2])

# sliding the same as list
print(tensorS) 
print(tensorS[2:4]) # ตัดมาแค่ 2 กับ 3

# assigning new values to a slide of tensor
print(tensorS.size()) # torch.Size([6])
tensorS[3:5] = torch.tensor([30, 40])
print(tensorS)  # tensor([ 1,  2,  3, 30, 40,  6])


# _____________________________________

# section 6 : operations

# adding operaiton
tensorP = tensorS + 1  # torch will automatically cast 1 onto every element of tensor
print(tensorP)   # tensor([ 2,  3,  4, 31, 41,  7])

# Hadamand product
tensorA = torch.tensor([1,2])
tensorB = torch.tensor([3,4])
print(tensorA * tensorB)

# dot product
print(torch.dot(tensorA, tensorB))  # tensor(11)
print(torch.dot(tensorA, tensorB).item())  # 11
# product of dot product is scalar.
# In torch, t is still a tensor of size 1 and ndimension 1


# _____________________________________

# section 7 : universal function to get insights from items in tensor

# get the mean value of all tensor elements
# (can only use with float dtype)
# print(tensorA.mean())  will cause error cuz tensorA is an IntTensor.
print(tensorA.type(torch.FloatTensor).mean())  # tensor(1.5000)
print(tensorA.type(torch.FloatTensor).mean().item())  # 1.5

# get the max item out of tensor
print(tensorA.max())  # tensor(2)
print(tensorA.max().item())  # 2

# find sine of each item
print(torch.sin(tensorA))  # tensor([0.8415, 0.9093])

# creating linespace
tensorL = torch.linspace(-2, 2, steps=5)  # steps = จะเอาทั้งหมดกี่ตัว
print(tensorL)  # tensor([-2., -1.,  0.,  1.,  2.])


# ploting a sine graph using torch and matplotlib
x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# _____________________________________

# section 8 : clarification between size(5) and size(5, 1)
# the 1st one is 1d, so there is no concept of row and column
# but the second one is 2d, so there is 5 rows and 1 column

# Practice: convert the following tensor to a tensor object with 1 row and 5 columns

your_tensor = torch.tensor([1, 2, 3, 4, 5])

your_new_tensor = your_tensor.view(1, 5)
print("Original Size: ", your_tensor, your_tensor.size())
print("Size after view method", your_new_tensor)  # tensor([[1, 2, 3, 4, 5]])
print(your_new_tensor.ndimension(), your_new_tensor.size())  # 2  torch.Size([1, 5])

your_tensor2 = your_tensor.view(5, 1)  
print(your_tensor2, your_tensor2.size())  
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5]])