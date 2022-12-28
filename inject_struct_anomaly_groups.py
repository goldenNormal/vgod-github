# -*- ecoding: utf-8 -*-
# @Author: HuangYiHong
# @Time: 2022-06-19 14:18

import torch_geometric.utils as utils
import numpy as np
import torch
import random
from torch_geometric.datasets import AttributedGraphDataset as AGD

def str_ano_inject(dataset_str,outlier_part_rate=0.01):

    outlier_rate = outlier_part_rate
    print('Loading data: {}...'.format(dataset_str))

    root = './AttributesDataset'
    dataset = AGD(root=root,name=dataset_str)[0]

    attribute_dense,edges = dataset.x,dataset.edge_index
    try:
        attribute_dense = attribute_dense.to_dense()
    except:
        pass

    attribute_dense ,edges = attribute_dense.detach().numpy(),edges.detach().numpy()
    num_node = attribute_dense.shape[0]
    ori_num_edge = edges.shape[1]

    mList = np.asarray([3,5,10,15]).astype(int)

    part_cnt = int(num_node * outlier_rate)
    nList = np.ceil(part_cnt / np.asarray(mList)).astype(int)
    total = int(np.sum(mList * nList))
    num_part = mList.shape[0]

    print('Done. \n')

    # Random pick anomaly nodes
    all_idx = list(range(num_node))

    random.shuffle(all_idx)

    anomaly_idx = all_idx[:total]
    ano_label = np.zeros((num_node),dtype=np.uint8)
    for i in range(num_part):
        m = mList[i]
        n = nList[i]
        ano_idx = anomaly_idx[m*n*i:m*n*(i+1)]
        ano_label[ano_idx] = i+1

    # Disturb structure
    edges = edges.tolist()
    print('Constructing structured anomaly nodes...')
    for part_id in range(num_part):
        n ,m = nList[part_id],mList[part_id]
        for n_ in range(n):
            ano_idx = (ano_label==(part_id+1)).nonzero()[0]
            current_nodes = ano_idx[n_*m:(n_+1)*m]
            for i in current_nodes:
                for j in current_nodes:
                    edges[0].append(i)
                    edges[1].append(j)
    edge_index = torch.LongTensor(edges)
    edge_index = utils.remove_self_loops(utils.coalesce(edge_index))[0]
    import scipy.sparse as sp
    adj = sp.csr_matrix(utils.to_dense_adj(edge_index).squeeze().numpy())
    edges = (edge_index).detach().numpy()

    num_add_edge = edge_index.shape[1] - ori_num_edge
    print('Done. ({:.0f} edges are added) ,num_add_edge/ori_num_dge = {:.2f}'.format(num_add_edge,num_add_edge/ori_num_edge))

    print('Saving file...')

    import os
    import scipy.io as sio
    savedir = './struct_datasets'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    sio.savemat('{}/{}_adj.mat'.format(savedir,dataset_str), \
                {'Network': adj, 'Label': ano_label, 'Attributes': attribute_dense})
    sio.savemat('{}/{}_str.mat'.format(savedir,dataset_str), \
                {'Edge': edges, 'Label': ano_label, 'Attributes': attribute_dense})

if __name__ == '__main__':
    datasets = ['Cora','Citeseer','PubMed','Flickr']
    for data in datasets:
        str_ano_inject(data,outlier_part_rate=0.02)
        print('finish {}'.format(data))
