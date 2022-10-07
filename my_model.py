import torch
from torch import Tensor, nn
from torch_geometric.nn import GIN, MessagePassing, GAT, GATConv, GINConv,GCN,MLP
from torch_geometric.typing import OptTensor, OptPairTensor



class MeanConv(MessagePassing):

    def __init__(
            self,
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index,
                edge_weight: OptTensor = None, size: torch.Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        return out

    def message(self, x_j: Tensor,x_i:Tensor) -> Tensor:
        return x_j

class CovConv(MessagePassing):

    def __init__(
            self,
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index,ner_mean,
                edge_weight: OptTensor = None, size: torch.Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, ner_mean = ner_mean[edge_index[1]],edge_weight=edge_weight,
                             size=size)

        out = torch.sum(out,dim=-1)
        return out

    def message(self, x_j: Tensor,ner_mean) -> Tensor:

        return (x_j - ner_mean)**2

class NeiVar(nn.Module):
    def __init__(self,input_dim,emb_dim):
        super(NeiVar, self).__init__()
        # self.lin = MLP([input_dim,emb_dim,emb_dim],act='relu',dropout=0.2)
        self.lin = nn.Linear(input_dim,emb_dim)
        # self.GNN = GIN(input_dim,emb_dim,2,act=nn.LeakyReLU())
        self.mean = MeanConv()
        self.cov = CovConv()

    def forward(self,x,edge_index):
        h = self.lin(x)
        h = h/(torch.norm(h,dim=-1).reshape(-1,1))
        mean = self.mean(h,edge_index)
        var = self.cov(h,edge_index,mean)

        return var

class Recon(nn.Module):
    def __init__(self,input_dim,emb_dim,GNN):
        super(Recon, self).__init__()
        # self.lin = MLP([input_dim,emb_dim,emb_dim],act='relu',dropout=0.2)
        self.lin = nn.Linear(input_dim,emb_dim)
        self.gnn = GNN(emb_dim,emb_dim,2,input_dim)
        # self.lin2 = nn.Linear(emb_dim,input_dim)

    def forward(self,x,edge_index):
        h = self.lin(x)
        h = h/(torch.norm(h,dim=-1).reshape(-1,1))
        recon_x = self.gnn(h,edge_index)
        # recon_x = self.lin2(recon_x)
        return torch.sum(torch.square(x - recon_x),dim=-1)


