#2021/4/4
import torch
#Tensor dtype and API

#Specifies the data type  and possible values of the tensor.
#For example. torch.float32-> 32-bit floating-point
#Or torch.bool ->Boolean

#And the default data type for tensors is 32-bit floating-point

#If creating tensors with integers as arguments, it will create a 64-bit integer tensor by default.

print(torch.tensor([1,1]).dtype) #gives me torch.int64

#Specify the dtype

double_points = torch.ones(10,2,dtype=torch.double)
short_points = torch.tensor([[1,2], [3,4]], dtype=torch.short)  #torch.int16

print(torch.tensor([1,1], dtype=torch.float).dtype) #gives me torch.float32

#use casting functions to make the tensors to be the expected type

double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)

#type of tensor will be converted to larger type when mixing input types in operations

print((torch.rand(5, dtype=torch.double) * torch.ones(5).short()).dtype) #gives me torch.float64

#The tensor API

#operations of tensor objects

a = torch.ones(3,2)
a_t = torch.transpose(a,0,1)
print(a.shape, a_t.shape)
#or
print(a.transpose(0,1).shape)

#Pointwise ops, take abs and cos functions as examples

a = torch.rand(3,2)
#print(a, a.abs(), a.cos())

#Reduction ops like mean, std and norm

print(a, a.mean(), a.std(), a.norm()) #overall mean, atd, norm

#For each part I can do

print(a.mean(-1))

##Comparison operations
print(a.equal(torch.randn(3,2)))
print(a.max())

#There are a lot of other operations and I will practice them while learning more about tensor