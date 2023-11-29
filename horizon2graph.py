import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import dgl
from datetime import datetime
import torch

merged_df = pd.read_csv('merged_horizon.csv', sep=',', on_bad_lines='skip')#data.head()

merged_df = merged_df[merged_df['status'] != 'SIGNED']
merged_df = merged_df.drop(columns=['nature'])
merged_df = merged_df.dropna()
merged_df['duration'] = (pd.to_datetime(merged_df['endDate']) - pd.to_datetime(merged_df['startDate'])).dt.days

features = merged_df[['totalCost_x', 'ecMaxContribution', 'legalBasis', 'topics', 'frameworkProgramme', 'masterCall', 'subCall', 'organisationID', 'SME', 'activityType', 'role', 'netEcContribution', 'totalCost_y', 'duration']]
def convert_to_float(x):
    if isinstance(x, str):
        return float(x.replace(',', '.'))
    return x

numeric_columns = ['totalCost_x', 'ecMaxContribution', 'netEcContribution', 'duration']
for col in numeric_columns:
    features[col] = features[col].apply(convert_to_float)
numeric_features = features[['totalCost_x', 'ecMaxContribution', 'netEcContribution', 'duration']]

categorical_features = features[['legalBasis', 'topics', 'frameworkProgramme', 'masterCall', 'subCall', 'organisationID', 'SME', 'activityType', 'role']]

label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_columns = ['totalCost_x', 'ecMaxContribution', 'netEcContribution', 'duration']
numeric_features = scaler.fit_transform(numeric_features)

encoded_features = pd.concat([pd.DataFrame(numeric_features, columns=numeric_columns), categorical_features.reset_index(drop=True)], axis=1)
features_tensor = torch.tensor(encoded_features.values, dtype=torch.float32)
labels = merged_df['status'].map({'CLOSED': 0, 'TERMINATED': 1})
labels_tensor = torch.tensor(labels.values)


g = dgl.DGLGraph()
g.add_nodes(merged_df.shape[0])
g.ndata['features'] = features_tensor
g.ndata['label'] = labels_tensor
organisation_dict = {}
for i, org_id in enumerate(encoded_features['organisationID']):
    if org_id not in organisation_dict:
        organisation_dict[org_id] = []
    organisation_dict[org_id].append(i)

from tqdm import tqdm
import random

for org_id, nodes in tqdm(organisation_dict.items()):
    src_nodes = []
    dst_nodes = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            src_nodes.append(nodes[i])
            dst_nodes.append(nodes[j])
            src_nodes.append(nodes[j])
            dst_nodes.append(nodes[i])
    total_edges = len(src_nodes)
    
    num_edges_to_keep = int(total_edges * 0.3)
    
    indices_to_keep = random.sample(range(total_edges), num_edges_to_keep)
    
    src_nodes = [src_nodes[i] for i in indices_to_keep]
    dst_nodes = [dst_nodes[i] for i in indices_to_keep]
    
    g.add_edges(src_nodes, dst_nodes)
import numpy as np


print(g)
dgl.save_graphs('horizon_x', g)
import dgl
import torch
import numpy as np
import os
import random
import pandas
# import bidict
from dgl.data import FraudAmazonDataset, FraudYelpDataset
from sklearn.model_selection import train_test_split
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
class Dataset:
    def __init__(self, name='tfinance', homo=True, add_self_loop=True, to_bidirectional=False, to_simple=True):
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
            graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
            graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
            graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
            graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
            graph.ndata['mark'] = graph.ndata['train_mask']+graph.ndata['val_mask']+graph.ndata['test_mask']
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask', 'mark'])

        else:
            graph = dgl.load_graphs('/home/yuhanli/wangpeisong/GADBench/horizon_x')[0][0]
        graph.ndata['feature'] = graph.ndata['features'].float()
        graph.ndata['label'] = graph.ndata['label'].long()
        self.name = name
        self.graph = graph
        if add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        if to_bidirectional:
            self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)
        if to_simple:
            self.graph = dgl.to_simple(self.graph)

    def split(self, samples=2):
        labels = self.graph.ndata['label']
        n = self.graph.num_nodes()
        if 'mark' in self.graph.ndata:
            index = self.graph.ndata['mark'].nonzero()[:,0].numpy().tolist()
        else:
            index = list(range(n))
        train_masks = torch.zeros([n,20]).bool()
        val_masks = torch.zeros([n,20]).bool()
        test_masks = torch.zeros([n,20]).bool()
        if self.name in ['tolokers', 'questions']:
            train_ratio, val_ratio = 0.5, 0.25
        if self.name in ['tsocial', 'tfinance', 'reddit', 'weibo']:
            train_ratio, val_ratio = 0.4, 0.2
        if self.name in ['amazon', 'yelp', 'elliptic', 'dgraphfin']:  # official split
            train_masks[:,:10] = self.graph.ndata['train_mask'].repeat(10,1).T
            val_masks[:,:10] = self.graph.ndata['val_mask'].repeat(10,1).T
            test_masks[:,:10] = self.graph.ndata['test_mask'].repeat(10,1).T
        else:
            train_ratio, val_ratio = 0.7, 0.1
            for i in range(10):
                seed = 3407+10*i
                set_seed(seed)
                idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*val_ratio), random_state=seed, shuffle=True)
                train_masks[idx_train,i] = 1
                val_masks[idx_valid,i] = 1
                test_masks[idx_test,i] = 1

        # for i in range(10):
        #     pos_index = np.where(labels == 1)[0]
        #     neg_index = list(set(index) - set(pos_index))
        #     pos_train_idx = np.random.choice(pos_index, size=2*samples, replace=False)
        #     neg_train_idx = np.random.choice(neg_index, size=8*samples, replace=False)
        #     train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4*samples]])
        #     train_masks[train_idx, 10+i] = 1
        #     val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4*samples:]])
        #     val_masks[val_idx, 10+i] = 1
        #     test_masks[index, 10+i] = 1
        #     test_masks[train_idx, 10+i] = 0
        #     test_masks[val_idx, 10+i] = 0

        self.graph.ndata['train_masks'] = train_masks
        self.graph.ndata['val_masks'] = val_masks
        self.graph.ndata['test_masks'] = test_masks

for data_name in ['horizon']:
    data = Dataset(data_name)
    data.split()
    print(data.graph)
    print(data.graph.ndata['train_masks'].sum(0), data.graph.ndata['val_masks'].sum(0), data.graph.ndata['test_masks'].sum(0))
    dgl.save_graphs(data_name, [data.graph])

