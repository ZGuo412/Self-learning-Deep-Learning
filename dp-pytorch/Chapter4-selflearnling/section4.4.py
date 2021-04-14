#2021/04/15

##Dealing with data in a flat table.
import numpy as np
import torch, torchvision

bikes_numpy = np.loadtxt(
    "../Data/bike-sharing-dataset/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    converters={1: lambda x: float(x[8:10])})

bikes = torch.from_numpy(bikes_numpy)
#reshape the date based on the time
daily_bikes = bikes.view(-1, 24, bikes.shape[1])

daily_bikes = daily_bikes.transpose(1, 2) #transpose based on the 1,2 dimension
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
weather_onehot.scatter_(
    dim=1,
    index=first_day[:,9].unsqueeze(1).long() - 1,
    value=1.0)

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])

daily_weather_onehot.scatter_(
    1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0)

daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1) ##concentrate along C dimension

daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0  ##preprocess data to make it in the range 0 to 1

temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp))
                        / torch.std(temp))