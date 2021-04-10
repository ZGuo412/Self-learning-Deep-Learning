#2021/04/10

##Section4.1
##Images with tensor
import torch, torchvision
import imageio
import os

#Load the bobby image I used before in chapter 2

img_arr = imageio.imread('../bobby.jpg')  #numpy array-like object

print(img_arr.shape)  #720-> height; 1280->width; 3->color channels

##tensor requiresimage data with channel; height and width. In such case, I need to change the lay out

img = torch.from_numpy(img_arr)
out = img.permute(2,0,1)


##For multiple images, I want to save it in a batch

##set up the batch size
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

##load multiple images and store them in the batch
def img_layout(img):
    img_t = torch.from_numpy(img)
    img_t = img_t.permute(2,0,1)
    img_t = img_t[:3] ##to aviod the alpha channel just in case.
    return img_t

data_dir = '../Data/image-cats/'
filenames = [name for name in os.listdir(data_dir)] #only contains png files
for i,filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    batch[i] = img_layout(img_arr)

##caset the tensor and normalize it to make the input data ranges from 0 to 1

batch = batch.float()
batch /=255.0

###Besides divided by 255, I can also calculate the mand and std to normalize the data

#For example


n_channels = batch.shape[1]  #give me 3
for c in range(n_channels):
    mean = torch.mean(batch[:, c]) #calcualte the mean for each color channel
    std = torch.std(batch[:, c]) #calculate the std for each color channel
    batch[:, c] = (batch[:, c] - mean) / std #scaling  -> will give me 0 mean and unit std
