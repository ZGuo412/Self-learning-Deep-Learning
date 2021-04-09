#2021/04/09

#Exercise problems for chapter 3


import torch, torchvision
#Problem 1
a = torch.tensor(list(range(9)))

#guess: 9, 0 ,1 for size, offset and stride
print(a,a.size(),a.storage_offset(),a.stride())

b = a.view(3,3)
print(b)  ##resize b from a

print(b.storage())
a[1] = 2
print(b)
##a and b share the same storage

a[1] = 1
c= b[1:,1:] #size:2x2, offset is 4 and stride is 2,1
print(c, c.size(), c.storage_offset(), c.stride())  #actually stride is 3,1, I need to check 4 in b not c, that's why 3,1

#Problem 2
a = torch.tensor(list(range(9))).float()

b = a.cos()
print(b)   ##elementwise cos for a and it do not support for long, so I change the type to float

a.cos_()
print(a)  #in place cos