# -*- ecoding: utf-8 -*-
# @Author: HuangYiHong
# @Time: 2022-09-11 16:04
import scipy.io as sio
import numpy as np
import torch
import torch_geometric.utils as utils
from torch_geometric.data import Data

def load_pyg_data(d_name,path=r'/root/data'):
    data = sio.loadmat(f'{path}/{d_name}_str.mat')
    label = data['Label'].reshape(-1)
    attribute = torch.FloatTensor(data['Attributes'])
    edge = torch.LongTensor(data['Edge'])
    y = torch.LongTensor(label)
    pygData = Data(x=attribute,edge_index=edge,y=y)
    return pygData

def load_mat(d_name,path='./data'):
    data = sio.loadmat(f'{path}/{d_name}.mat')
    adj = torch.LongTensor(data['Network'].toarray())
    attr = torch.FloatTensor(data['Attributes'].toarray())
    label = torch.LongTensor(data['Label'].reshape(-1))
    str_label = torch.LongTensor(data['str_anomaly_label'].reshape(-1))
    attr_label = torch.LongTensor(data['attr_anomaly_label'].reshape(-1))
    edge_index = utils.dense_to_sparse(adj)[0]

    pygData = Data(x=attr,edge_index=edge_index,y=label,str_y=str_label,attr_y=attr_label)
    return pygData

from pygod.utils import load_data as ld
def load_weibo():
    data = ld(name='weibo')
    data.str_y = data.attr_y = data.y
    return data

if __name__ == '__main__':
    pass

    load_mat('cora')