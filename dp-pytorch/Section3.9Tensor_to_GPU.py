#section3.9-3.10
#2021/04/06

#create a tensor on the GPU
import torch
import numpy as np

#points_gpu = torch.tensor([[4,1],[5,3], [2,1]], device='cuda')

#I can use points.to(device = 'cuda')

points = torch.tensor([[4,1],[5,3], [2,1]])
points = 2 * points ##on CPU

#points_gpu = 2 * points.to(device = 'cuda')

##Other function .cuda() / .cpu()

##Section 3.10 about numpy interoperability

points = torch.ones(3,4) #Create a tensor array with dimension 3*4

points_np = points.numpy() ##convert tensor to numpy

points = torch.from_numpy(points_np) #convert it back to tensor array
