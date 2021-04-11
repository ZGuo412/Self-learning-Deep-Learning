#2021/04/11
##Section 4.2 about loading images with a specialized format


import imageio, torch, torchvision


#Using volread function to read images in dirand assemble all files like(DICOM) Medicine files in numpy array
dir_path = "../Data/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')

print(vol_arr.shape)   ##There is no channel information about the image data

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol,0) #add one more dimension to the vol

