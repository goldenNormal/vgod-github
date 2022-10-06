import scipy.io as sio
import torch
from torch import nn
from torch_geometric.data import Data
import argparse
import torch_geometric.utils as utils
from torch.optim import Adam

from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCN

from EarlyStop import entropy_gradient
from transform import NormalizeToOne

def load_pyg_data(d_name,path=r'/root/data'):
    data = sio.loadmat(f'{path}/{d_name}_str.mat')
    label = data['Label'].reshape(-1)
    attribute = torch.FloatTensor(data['Attributes'])
    edge = torch.LongTensor(data['Edge'])
    y = torch.LongTensor(label)
    pygData = Data(x=attribute,edge_index=edge,y=y)
    return pygData

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='Flickr')

args = parser.parse_args()

data = load_pyg_data(args.data,path='./struct_datasets')# in PyG format


print(f'finish load {args.data}')
y = data.y.cpu().numpy() >0  # binary labels (inl
edge_index = data.edge_index

deg = utils.degree(edge_index[1],num_nodes=data.x.size(0),dtype=data.x.dtype)

score = deg.detach().numpy()
import numpy as np
np.save(f'./results/deg_{args.data}_str.npy',score)
