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