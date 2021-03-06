#2021/4/5
#Section3.8

###size: size of tensor
###offset:return the index of the first element in the whole storage
import torch
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
#print(second_point)
#5 in the storage is in the index 2
print(second_point.storage_offset())

##Basically same information saved in size and shape functions
print(points.size(), points.shape)


#+1 for next col and +2 for next row
print(points.stride()) ##gives me (2,1) which refers to the num of elements need to be skipped in storage

#For any element in tensor i,j tensor[i][j] = tensor[offset + i*stride[0] + j*stride[1]]

print(points[2][1] == (points.storage()[points.storage_offset() + 2*points.stride()[0] + points.stride()[1]]))

second_point = points[1]
print(second_point.size())
print(second_point.storage_offset())
print(second_point.stride())

#Since it index in same storage, modify second point will change the value in points as well
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point[0] = 10.0
print(points)

#By clone, it will not

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
print(points)


###Transpose function:

#Here I tried to use an in-place transpose, both points share the same storage
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.t_()
print(points)

#For stride they are different

print(points.stride() == points.t().stride())

###For multi-dimension tranpose

some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)  #Transpose first and third dimension

transpose_tt = some_t.transpose(0,1) #Transpose first and second dimension

print(transpose_t.shape, transpose_tt.shape)

###Contiguous tensors: check if the order of values managed by the storage is same as order for values moving from rows.

#view only works for contiguous tensors

print(points.is_contiguous())
print(points.t().is_contiguous())

#it is able to get a contiguous tensor from uncontiguous tensor

point_c = points.contiguous()
print(point_c.is_contiguous(), point_c)
