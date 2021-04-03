#2021/04/02
import torch


#initialize tensor array with len=6 and each element equals to zero
points = torch.zeros(6)


#Pass a list to tensor
list_p = [4,3,2,1,1,2,3]
points = torch.tensor(list_p)
#print(type(points), points)

#2D tensor

points = torch.tensor([[4,1], [5,3], [2,1]])
#print(points.shape, points)

#3*2 all zero 2d tensor

points = torch.zeros(3,2)

#access an individual element with indices
points[0,0]

#About the index
s_list = torch.rand(6)
#print(s_list)

s_list[:] ##The whole tensor list
s_list[1:4] #elements from 1 to 4 exclusive
s_list[1:] #elements 1 to end
s_list[:] #start to 4 exclusive
s_list[:-1] #except last element
#print(s_list[1:4:2]) #take elements with step 2

s_list = torch.rand(3,3)
print(s_list, s_list[1:]) #all the rows after the first one

print(s_list[1:, :]) #same as first one

s_list[1:,0] #all the rows with first column

print(s_list[None]) #Add dimension of size 1 like unsqueeze 0