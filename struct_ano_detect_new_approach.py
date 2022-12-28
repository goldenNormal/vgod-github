from  my_model import NeiVar
from torch.optim import Adam
import torch
import torch_geometric.utils as utils
from sklearn.metrics import roc_auc_score
from transform import standScale
import numpy as np

from torch_geometric.transforms import NormalizeFeatures
from sklearn.preprocessing import StandardScaler
import argparse
from load_data import load_mat
from torch_geometric.nn import GIN,GAT
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:1000"

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='pubmed')

parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--lr', type=float, default=0.005)
# learning rate for datasets : 0.005
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = ['cora','citeseer','pubmed','Flickr']
# datasets = ['cora']
for data_str in datasets:
    args.data = data_str
    data = load_mat(args.data+"_novelstr",path='./NovelStr_datasets')
    lr = 0.005

    y = data.y.numpy()

    import torch_geometric.utils as U
    print(data_str)
    deg = U.degree(data.edge_index[0],num_nodes=data.x.shape[0])
    print('deg score',roc_auc_score(data.y.numpy(),deg.numpy()))

    data = data.to(device)

    print(f'finish load {args.data}')
    # y = (data.y==1).cpu().numpy()   # binary labels (inl
    edge_index = data.edge_index

    if args.data != 'Flickr':
        edge_index = utils.add_self_loops(edge_index)[0]
        # pass
    else:
        data = standScale(data.to('cpu')).to(device)

    # Adj = utils.to_dense_adj(edge_index).squeeze()

    input_dim = data.x.size(1)
    emb_dim = 128
    lr =lr
    num_epoch = args.epoch
    alpha = 1

    # model = CooTrain(input_dim,emb_dim).to(device)
    struct_model = NeiVar(input_dim,emb_dim).to(device)
    str_opt = Adam(struct_model.parameters(),lr=lr,weight_decay=0.0001)

    @torch.no_grad()
    def eval_model():
        global  y
        struct_model.eval()
        score_var = struct_model(data.x,edge_index)
        return roc_auc_score(y,score_var.detach().cpu().numpy())


    def loss_var_fn(pos_loss,neg_loss):
        return torch.mean(pos_loss) - torch.mean(neg_loss)

    def train():
        struct_model.train()
        neg_edge = utils.negative_sampling(edge_index,num_neg_samples=edge_index.size(1) )
        pos_loss = struct_model(data.x,edge_index)
        neg_loss= struct_model(data.x,neg_edge)
        var_loss = loss_var_fn(pos_loss,neg_loss)


        str_opt.zero_grad()
        var_loss.backward()
        str_opt.step()
        return float(var_loss)

    print('begin train')
    for e in range(num_epoch):
        train_loss = train()
        test_auc = eval_model()
        print(f'Epoch: {e}, auc:{test_auc},')

