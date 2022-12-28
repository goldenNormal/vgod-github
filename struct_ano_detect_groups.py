import pygod
from  my_model import NeiVar,Recon
from torch.optim import Adam
import torch
import torch_geometric.utils as utils
from pygod.utils import load_data
from sklearn.metrics import roc_auc_score
from transform import NormalizeToOne,standScale,minMaxScale
from load_data import load_pyg_data
import argparse
import numpy as np
from torch_geometric.transforms import NormalizeFeatures
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='PubMed') # 'Cora' 'Citeseer' 'PubMed' 'Flickr'
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_pyg_data(args.data,path='./struct_datasets')# in PyG format

# trans = NormalizeFeatures()
# data = trans(data)
# data = NormalizeToOne(data)

data = data.to(device)

print(f'finish load {args.data}')
y = data.y.cpu().numpy()   # binary labels (inl

edge_index = data.edge_index
# edge_index,_ = utils.add_self_loops(edge_index)

input_dim = data.x.size(1)
emb_dim = 128
lr = 0.005
if args.data == 'Citeseer':
    num_epoch = 3
elif args.data == 'Cora' or args.data =='PubMed':
    num_epoch = 4
else:
    num_epoch = 15

model = NeiVar(input_dim,emb_dim).to(device)
opt = Adam(model.parameters(),lr=lr,weight_decay=0.0001)
Entro = []

@torch.no_grad()
def eval_model():
    global  y
    model.eval()
    score = model(data.x,edge_index).cpu().detach().numpy()
    Auc = []
    for i in range(1,y.max()+1):
        part_label = (y==i).astype(int)
        part_label = part_label.reshape(score.shape)
        auc = roc_auc_score(part_label,score)
        Auc.append(auc)

    return Auc,roc_auc_score(y>0,score),score

def loss_var_fn(pos_loss,neg_loss):
    return torch.mean(pos_loss) - torch.mean(neg_loss)

def train():
    model.train()
    neg_edge = utils.negative_sampling(edge_index,num_neg_samples=edge_index.size(1) )

    pos_loss = model(data.x,edge_index)
    neg_loss= model(data.x,neg_edge)
    loss = loss_var_fn(pos_loss,neg_loss)


    opt.zero_grad()
    loss.backward()
    opt.step()

    return float(loss)

print('begin train')

min_entro = -1

for e in range(num_epoch):
    train_loss = train()
    test_auc,auc,score = eval_model()

    print(f'Epoch: {e}, trainLoss: {train_loss:.2f}, Aucs: ',end='')
    for i in range(len(test_auc)):
        print(f'{test_auc[i]:.4f},',end='')
    print(f'AllAuc:{auc:.4f}')


import matplotlib.pyplot as plt

# output score
with torch.no_grad():
    model.eval()
    score = model(data.x,edge_index).cpu().detach().numpy()
    np.save(f'./results/{args.data}_str.npy',score)

# plt.subplot(1,2,1)
# plt.plot(Entro)
# plt.subplot(1,2,2)
# plt.plot(np.gradient(Entro,1))
# plt.show()