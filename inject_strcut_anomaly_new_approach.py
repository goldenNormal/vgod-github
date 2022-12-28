import numpy as np
import scipy.sparse as sp
import random
import scipy.io as sio
import argparse
import pickle as pkl
import networkx as nx
import sys
import os
import os.path as osp
from sklearn import preprocessing
from scipy.spatial.distance import euclidean,cosine

def dense_to_sparse(dense_matrix):
    shape = dense_matrix.shape
    row = []
    col = []
    data = []
    for i, r in enumerate(dense_matrix):
        for j in np.where(r > 0)[0]:
            row.append(i)
            col.append(j)
            data.append(dense_matrix[i,j])

    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return sparse_matrix

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_datadet(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("raw_dataset/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("raw_dataset/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj_dense = np.array(adj.todense(), dtype=np.float64)
    attribute_dense = np.array(features.todense(), dtype=np.float64)
    cat_labels = np.array(np.argmax(labels, axis = 1).reshape(-1,1), dtype=np.uint8)

    return attribute_dense, adj_dense, cat_labels



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed'
parser.add_argument('--seed', type=int, default=100)  #random seed
parser.add_argument('--k', type=int, default=50)  #num of clusters
args = parser.parse_args()


AD_dataset_list = ['BlogCatalog', 'Flickr']
Citation_dataset_list = ['cora', 'citeseer', 'pubmed']

# Set hyperparameters of disturbing
dataset_str = args.dataset  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed'
seed = args.seed
k = args.k

if __name__ == "__main__":

    # Set seed

    datas = ['cora','citeseer','pubmed','Flickr']
    # Load data
    for dataset_str in datas:
        print('Random seed: {:d}. \n'.format(seed))
        np.random.seed(seed)
        random.seed(seed)

        print('Loading data: {}...'.format(dataset_str))
        if dataset_str in AD_dataset_list:
            data = sio.loadmat('./raw_dataset/{}/{}.mat'.format(dataset_str, dataset_str))
            attribute_dense = np.array(data['Attributes'].todense())
            # attribute_dense = preprocessing.normalize(attribute_dense, axis=0)
            attribute_dense = preprocessing.scale(attribute_dense,axis=0)
            adj_dense = np.array(data['Network'].todense())
            cat_labels = data['Label']
        elif dataset_str in Citation_dataset_list:
            attribute_dense, adj_dense, cat_labels = load_citation_datadet(dataset_str)

        ori_num_edge = np.sum(adj_dense)
        num_node = adj_dense.shape[0]
        print('Done. \n')

        # Random pick anomaly nodes
        all_idx = list(range(num_node))
        random.shuffle(all_idx)
        ano_cnt = int(num_node * 0.1)
        anomaly_idx = all_idx[:ano_cnt]
        structure_anomaly_idx = anomaly_idx
        # attr_noise_idx = anomaly_idx[part_cnt:]

        # attribute_anomaly_idx = anomaly_idx[m*n:]
        label = np.zeros((num_node,1),dtype=np.uint8)
        label[anomaly_idx,0] = 1

        str_anomaly_label = label


        # Disturb structure
        print('Constructing structured anomaly nodes...')
        for node_id in structure_anomaly_idx:
            cat_y = cat_labels[node_id]
            all_other_label_nodes = (cat_labels.reshape(-1)!=cat_y).nonzero()[0]
            ori_deg = int(np.sum(adj_dense[node_id]))
            new_neibors = np.random.choice(all_other_label_nodes,ori_deg,replace=False)
            adj_dense[node_id]= 0
            adj_dense[node_id,new_neibors] = 1
            adj_dense[new_neibors,node_id] = adj_dense[node_id,new_neibors].copy()

        # Pack & save them into .mat
        print('Saving mat file...')
        attribute = dense_to_sparse(attribute_dense)
        adj = dense_to_sparse(adj_dense)

        savedir = './NovelStr_datasets/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        sio.savemat(f'{savedir}/{dataset_str}_novelstr.mat', \
                    {'Network': adj, 'Label': str_anomaly_label, 'Attributes': attribute, \
                     'Class':cat_labels, 'str_anomaly_label':str_anomaly_label, 'attr_anomaly_label':str_anomaly_label,
                     })
        print('Done. The file is save as: dataset/{}.mat \n'.format(dataset_str))


