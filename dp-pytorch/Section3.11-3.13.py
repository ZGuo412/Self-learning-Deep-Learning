##Rest of the section 3
##2021/04/07


##Serializing tensors

#save points into .t file

import torch, torchvision
import h5py


points = torch.tensor([[4,1], [3,2]])

torch.save(points, 'Data/points_out.t')

##or

with open('Data/points_out.t', 'wb') as f:
    torch.save(points, f)


##Load back the points data

points = torch.load('Data/points_out.t')

#or

with open('Data/points_out.t', 'rb') as f:
    points = torch.load(f)

##save points as hdf5 file

f = h5py.File('Data/points_out.hdf5', 'w')

##coords is the key into the HDF5 file
##convert the points to numpy array and pass the data into create dataset function
dset = f.create_dataset('coords', data=points.numpy())
f.close()

#load the file and only use part of the data
f = h5py.File('Data/points_out.hdf5', 'r')
dset = f['coords']

part_points = dset[-1:]
print(part_points)

#convert the numpy data to tensor

part_points_tensor = torch.from_numpy(part_points)
f.close()

print(part_points_tensor, part_points_tensor.dtype, part_points_tensor.float().dtype)
