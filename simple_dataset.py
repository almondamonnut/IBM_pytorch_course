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
        