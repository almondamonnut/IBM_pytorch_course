# This code is demonstration of many useful codes
# to use with torch tensor 

import torch
import numpy as np

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

# type castiing
a = a.type(torch.FloatTensor)
print(a.type(), a.dtype)

b = a.type(torch.int32)  # ข้างในวงเล็บจะใส่เป็น dtype หรือ type ของ tensor ก็ได้
print(b.type(), b.dtype)

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