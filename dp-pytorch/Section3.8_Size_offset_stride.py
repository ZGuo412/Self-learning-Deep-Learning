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

