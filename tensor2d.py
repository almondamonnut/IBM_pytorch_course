# This code is to demonstrate the knowledge of torch tensor 2d
import torch
import pandas as pd

a = [[1,2], [3,4], [5,6]]
A = torch.tensor(a)

# to check the shape of a 2-d tensor
print(A.shape)   # torch.Size([3, 2])

# index and slicing
# Use tensor_obj[row, column] and tensor_obj[row][column] to access certain position
print(A)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
# first number = row number
# second number = column number
print(A[0][1])   # tensor(2)
print(A[0][1:20])  # tensor([2]) # the end point can be any number, even out of range still ok

# we can also do the comma seperation to specify the position of elements too
print(A[0, 1])  # # tensor(2)
print(A[1:3, 0])  # tensor([3, 5])

# matrix multiplication (mm)
# matrix multiplication can be done when number of columns of the 1st one equals number of rows of the second one
A = torch.tensor([[1,0,1],[0,1,0]])  # 3 columns
B = torch.tensor([[1],[0],[1]])      # 3 rows (1 columns)
print(torch.mm(A, B))
# tensor([[2],
#         [0]])



# convert the whole pandas df to tensor
df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
print(df)
# A  B
# 0  1  4
# 1  2  5
# 2  3  6

tensorD = torch.from_numpy(df.values)
print(tensorD)

# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])


# auto build tensor with torch.ones(row, column)
x = torch.ones(10, 2)
print(x, x.ndimension(), x.shape)