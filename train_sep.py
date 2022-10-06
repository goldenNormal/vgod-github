from  my_model import NeiVar,Recon,CooTrain
from torch.optim import Adam
import torch
import torch_geometric.utils as utils
from sklearn.metrics import roc_auc_score
from transform import NormalizeToOne,standScale,minMaxScale
import numpy as np

from torch_geometric.transforms import NormalizeFeatures
from sklearn.preprocessing import StandardScaler
import argparse
from load_data import load_mat,load_weibo
from torch_geometric.nn import GIN,GAT
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:1000"

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='weibo')
# parser.add_argument('--y', type=int, default=1)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--str-epoch', type=int, default=10)
# parser.add_argument('--lr', type=float, default=0.005)
# learning rate for datasets : weibo 0.01, else 0.005
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.data == 'weibo':
    data = load_weibo()
    lr = 0.01
else:
    data = load_mat(args.data)
    lr = 0.005

y = data.y.numpy()
str_y = data.str_y.numpy()
attr_y = data.attr_y.numpy()

if args.data == 'weibo':
    trans = NormalizeFeatures()
    data = trans(data)
    # pass
# data = NormalizeToOne(data)
# data = standScale(data)
print('norm attr score',roc_auc_score(data.attr_y.numpy(),torch.norm(data.x,dim=-1).numpy()))
print('norm score',roc_auc_score(data.y.numpy(),torch.norm(data.x,dim=-1).numpy()))

data = data.to(device)

print(f'finish load {args.data}')
# y = (data.y==1).cpu().numpy()   # binary labels (inl
edge_index = data.edge_index
# edge_index = utils.add_self_loops(edge_index)[0]

input_dim = data.x.size(1)
emb_dim = 128
lr =lr
num_epoch = args.epoch
alpha = 1

# model = CooTrain(input_dim,emb_dim).to(device)
struct_model = NeiVar(input_dim,emb_dim).to(device)
if args.data == 'weibo':
    GNN = GAT
else:
    GNN = GIN
context_model = Recon(input_dim,emb_dim,GNN).to(device)
opt = Adam(list(struct_model.parameters()) + list(context_model.parameters()),lr=lr,weight_decay=0.0001)

def add_two_score(score1,score2):
    score1 = score1/np.sum(score1)
    score2 = score2/np.sum(score2)
    return score1 + score2

def std_scale(x):
    x = (x - np.mean(x))/np.std(x)
    return x

def add_two_score_std(score1,score2):
    score1 = std_scale(score1)
    score2 = std_scale(score2)
    return score1 + score2 * alpha

@torch.no_grad()
def eval_model():
    global  y
    struct_model.eval()
    context_model.eval()
    score_recon= context_model(data.x,edge_index)
    score_var = struct_model(data.x,edge_index)
    score_recon,score_var =score_recon.cpu().detach().numpy(),score_var.cpu().detach().numpy()
    y = y.reshape(score_recon.shape)
    score = add_two_score_std(score_recon,score_var)

    return roc_auc_score(y,score),roc_auc_score(str_y,score), \
           roc_auc_score(attr_y,score)

def loss_recon_fn(recon_loss):
    return torch.mean(recon_loss)

def loss_var_fn(pos_loss,neg_loss):
    return torch.mean(pos_loss) - torch.mean(neg_loss)

def train(e):
    var_loss = 0
    if args.str_epoch > e:
        struct_model.train()
        neg_edge = utils.negative_sampling(edge_index,num_neg_samples=edge_index.size(1) )
        pos_loss = struct_model(data.x,edge_index)
        neg_loss= struct_model(data.x,neg_edge)
        var_loss = loss_var_fn(pos_loss,neg_loss)

    context_model.train()
    recon_loss = context_model(data.x,edge_index)
    recon_loss = loss_recon_fn(recon_loss)

    # loss = recon_loss + alpha * var_loss

    opt.zero_grad()
    recon_loss.backward()
    if args.str_epoch > e:
        var_loss.backward()
    opt.step()

    return float(recon_loss + alpha * var_loss)

print('begin train')
for e in range(num_epoch):
    train_loss = train(e)
    test_auc = eval_model()
    print(f'Epoch: {e}, trainLoss: {train_loss}, auc: {test_auc[0]:.3f}'
          f', str_auc: {test_auc[1]:.3f}, attr_auc: {test_auc[-1]:.3f}')

with torch.no_grad():
    struct_model.eval()
    context_model.eval()
    score_recon= context_model(data.x,edge_index)
    score_var = struct_model(data.x,edge_index)
    score_recon,score_var =score_recon.cpu().detach().numpy(),score_var.cpu().detach().numpy()
    score = add_two_score_std(score_recon,score_var)
    np.save(f'./results/mine_{args.data}',score)
