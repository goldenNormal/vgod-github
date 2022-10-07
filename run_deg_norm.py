from  my_model import NeiVar,Recon
from torch.optim import Adam
import torch
import torch_geometric.utils as utils
from sklearn.metrics import roc_auc_score
from transform import NormalizeToOne,standScale,minMaxScale
import numpy as np
from torch_geometric.transforms import NormalizeFeatures
from sklearn.preprocessing import StandardScaler
import argparse
from load_data import load_mat, load_weibo

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Cora')
args = parser.parse_args()


if args.data == 'weibo':
    data = load_weibo()
else:
    data = load_mat(args.data)
y = data.y.numpy()
str_y = data.str_y.numpy()
attr_y = data.attr_y.numpy()


# data = standScale(data)
print('norm attr score',roc_auc_score(data.attr_y.numpy(),torch.norm(data.x,dim=-1).numpy()))
print('norm score',roc_auc_score(data.y.numpy(),torch.norm(data.x,dim=-1).numpy()))


print(f'finish load {args.data}')
# y = (data.y==1).cpu().numpy()   # binary labels (inl
edge_index = data.edge_index
# edge_index = utils.add_self_loops(edge_index)[0]

deg = utils.degree(edge_index[1],num_nodes=data.x.size(0)).numpy()
norm = torch.norm(data.x,dim=-1).numpy()

def std_scale(x):
    x = (x - np.mean(x))/np.std(x)
    return x

def add_two_score_std(score1,score2):
    score1 = std_scale(score1)
    score2 = std_scale(score2)
    return score1 + score2

score = add_two_score_std(deg,norm)
aucs = [roc_auc_score(y,score),roc_auc_score(str_y,score),roc_auc_score(attr_y,score)]

print(f' auc: {aucs[0]:.3f}'
      f', str_auc: {aucs[1]:.3f}, attr_auc: {aucs[-1]:.3f}')

method = 'deg-norm'
np.save(f'./results/{method}_{args.data}',score)