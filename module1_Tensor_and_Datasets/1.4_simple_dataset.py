# This code is to demonstrate how to
# 1. Build dataset class & object
# 2. Build a Dataset Transform
# 3. Compose transforms 

import torch

# 1. Build dataset class & object

from torch.utils.data import Dataset

# สร้าง dataset class เป็น subclass ของ torch.utils.data.Dataset
class datakung(Dataset):
    
    def __init__ (self, length = 5, transform = None):
        self.length = length
        self.transform = transform
        self.len = length

        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
    
    # def a function to be show data_instance[index] as a row of sample at that index
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    # def a function to show len(data_instance)
    def __len__(self):
        return self.len

# create an instance of datakung class
data1 = datakung()
print(data1)
print(len(data1))   # 100
print(data1[3])     # (tensor([2., 2.]), tensor([1.]))

#___________________________________________



# 2. Build a Dataset Transform

class normalizee():

    def __init__(self):
        pass
        
    def __call__(self, sample):
        x, y = sample # x and y are both tensors
        print("x.mean().item():",x.mean().item())
        print("x.std().item():", x.std().item())
        nor_x = (x - x.mean().item()) / x.std().item()
        nor_y = (y - y.mean().item()) / y.std().item()    
        # กรณีนี้ข้อมูลเราไม่มี std (ค่าเท่ากันหมดทุกตัว) ก็เลยได้ nan
        return nor_x, nor_y
    
normalizekung = normalizee()
print(normalizekung(data1[:]))


class multiplyy():

    def __init__(self, mfactor):
        self.mfactor = mfactor
    
    # ใน __call__ ควรจะรับแค่ sample อย่างเดียว 
    # factor อื่นต้องอยู่ที่ __init__ ให้หมด
    # เพื่อให้ตอนเรียกใช้ผ่าน Compose ผ่าน Dataset subclass 
    # จะใส่แค่ sample เท่านั้น มาตรฐานเดียวกันทุก transform function
    def __call__(self, samplee):    
        x, y = samplee
        x *= self.mfactor
        y *= self.mfactor
        return x, y

multy = multiplyy(20)
print(multy(data1[:]))
# (tensor([[40., 40.],
#         [40., 40.],
#         [40., 40.],
#         [40., 40.],
#         [40., 40.]]), tensor([[20.],
#         [20.],
#         [20.],
#         [20.],
#         [20.]]))
        

#___________________________________________


# 3. Compose many transfoms together

from torchvision import transforms

# make a callable instance of class transforms.Compose 
# by passing a list of instances of transform classes
data_transform = transforms.Compose([normalizee(), multiplyy(20)])

# apply without specifying transform in Dataset subclass
x_, y_ = data_transform(data1[:])
print(x_, y_)

# instantiate Dataset subclass with transform specified
dataset_tr = datakung(transform = data_transform)
print(dataset_tr)
print(dataset_tr[0])
# (tensor([nan, nan]), tensor([nan]))