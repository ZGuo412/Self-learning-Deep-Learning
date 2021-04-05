#2021、04、05

##section 3.7 about tensor storage

##Values in tensors are managed by torch.storage, which is a one-dimensional array of numerical data(float or int)

##Multile tensors can index the same storage


import torch

points = torch.tensor([[4,1],[5,3],[-1,-2]])

print(points.shape)
print(points.storage())

#can also index into storage manually by points.storage()[n]

##Section 3.72 using in-place operations to change values stored in tensor


a = torch.ones(3,2)

a.zero_()

print(a)