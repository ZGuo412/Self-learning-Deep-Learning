#2021/4/3 named_tensor section 3.4

import torch, torchvision
img_t = torch.randn(3,5,5)  #refers to channel, rows and columns
weights = torch.tensor([0.2126, 0.7152, 0.0722]) #list to tensor array



batch_t = torch.randn(2,3,5,5) #refers to batch, channel, rows and columns

#get the unweighted mean from the image, batch of images data
#averafe of channel

img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)


##broadcasting
#same as unsqueezed_weights[None] twice and assgin the result to unsqueezed_weights
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(1)


img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)


#named channel part
#It is an experimental feature
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])

#For other named tensor, even I dont want to reinitialize them again, I need to use and practice align_as function......-.-
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')

#Aligned as function help me to check dimension and do broadcasting

weights_aligned = weights_named.align_as(img_named)
print(weights_aligned.shape)

