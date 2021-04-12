#2021/04/12
##Dealing with wine data



import csv
import torch
import numpy as np

wine_path = "../Data/tabular-wine/winequality-white.csv"

##load data as np

np_wine = np.loadtxt(wine_path, dtype=np.float32,delimiter=";", skiprows=1)  #The first row contains the names

col_list = next(csv.reader(open(wine_path),delimiter=";"))

##convert the np to tensor

tensor_wine = torch.from_numpy(np_wine)

#Make the quality as the ground truth

input_data = tensor_wine[:,:-1]

target = tensor_wine[:,-1]

label = target.long()
#print(label.dtype)

######Method of one hot encoding

target_onehot = torch.zeros(label.shape[0], 10)

target_onehot.scatter_(1, label.unsqueeze(1), 1.0)  ##add dimension to make it with the same dimension of target_honehot


print(input_data.shape)

##get the mean of the data

data_mean = torch.mean(input_data, dim=0)
data_var = torch.var(input_data, dim=0)

##normalize the data

data_norm = (input_data - data_mean) / torch.sqrt(data_var)

bad_indexes = label <=3
print(bad_indexes)  #gives me false or true->bool

bad_data = input_data[bad_indexes]
#print(bad_data)

bad_data = input_data[label <= 3]
mid_data = input_data[(label > 3) & (label <7)]
good_data = input_data[label >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))


#The book use total sulfur dioxide as the threshold and I try to try more.


#First total sulfur dioxide

total_sulfur_threshold = 141.83
total_sulfur_data = input_data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
actual_indexes = target > 5

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches, n_matches / n_predicted, n_matches / n_actual)

##If I use fixed acidity

acidity_threshold = 6.89
acidity_data = input_data[:,1]
predicted_indexes = torch.lt(acidity_data, acidity_threshold)
actual_indexes = target > 5


n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches, n_predicted, n_matches / n_predicted, n_matches / n_actual)