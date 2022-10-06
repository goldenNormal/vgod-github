# -*- ecoding: utf-8 -*-
# @Author: HuangYiHong
# @Time: 2022-09-08 11:25

import torch

from sklearn.preprocessing import StandardScaler,MinMaxScaler

def NormalizeToOne(data):
    x = data.x/(torch.norm(data.x,dim=-1).reshape(-1,1)+1e-10)
    data.x = x
    return data

def standScale(data):
    x = data.x.numpy()
    enc = StandardScaler()
    x = torch.from_numpy(enc.fit_transform(x)).type(torch.FloatTensor)
    data.x = x
    return data

def minMaxScale(data):
    x = data.x.numpy()
    enc = MinMaxScaler()
    x = torch.from_numpy(enc.fit_transform(x)).type(torch.FloatTensor)
    data.x = x
    return data

