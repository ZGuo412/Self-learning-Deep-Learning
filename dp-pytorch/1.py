##2021/03/29
##self learning pytorch and deep learning based on the book "Deep Learning with PyTorch"
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
##Check actual pretrained models

dir(models)

#Try AlexNet

alexnet = models.AlexNet()

#Try resnet101 model which trained on the ImageNet dataset

resnet = models.resnet101(pretrained=True)

#Look at the structure of the model
#print(resnet)


#Define a preprocess function
preprocess = transforms.Compose(
    [transforms.Resize(256),  ###scale the image to 256 * 256
     transforms.CenterCrop(224),  ###Crop the image to 224 * 224
     transforms.ToTensor(), ###Convert the imput data to tensor array
     transforms.Normalize(mean=[0.485, 0.456, 0.406],  ##Normalize the RGB components
                          std= [0.229, 0.224, 0.225])]
)


##Load the sample image
img = Image.open("cat.jpg")
##Show the image

#img.show()

##Pass the image through the preprocessing pipeline:
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

##put the network in eval mode to running the trained model

resnet.eval()

out = resnet(batch_t)

#I want to know the size of my out
#print(out.size())

##load the labels for the dataset calsses

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

##find out the index in out which has the highest score corresponding to the label


#I only care about the index.
_, index = torch.max(out, 1)

#index is not a number but a tensor array
#print(index)

###normalize outpus to the range [0,1] and divide by the sum
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()
_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:8]])



##Sort the array and get the indices of sorted values in the original array, which can show me the predicted results in descending order.
