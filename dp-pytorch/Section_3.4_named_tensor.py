#2021/4/3 named_tensor section 3.4

import torch, torchvision
img_t = torch.randn(3,5,5)  #refers to channel, rows and columns
weights = torch.tensor([0.2126, 0.7152, 0.0722]) #list to tensor array



batch_t = torch.randn(2,3,5,5) #refers to batch, channel, rows and columns

#get the unweighted mean from the image, batch of images data
#averafe of channel
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)

print(img_t, img_gray_naive)





