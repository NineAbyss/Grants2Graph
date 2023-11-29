import pandas as pd
import torch
import dgl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn

from tqdm import tqdm
import random

file_path = 'your path'
data = pd.read_csv(file_path, low_memory=False)
data = data[data['project_type'].isin(['面上项目', '青年科学基金项目', '地区科学基金项目'])]
data= data.query('end_time < 201701')

data[['subject_code_1', 'subject_code_2', 'subject_code_3']] = data['subject_code_list'].str.split('，', expand=True)

data['subject_code_1'] = data['subject_code_1'].str.replace('一级：', '')
data['subject_code_2'] = data['subject_code_2'].str.replace('二级：', '')
data['subject_code_3'] = data['subject_code_3'].str.replace('三级：', '')

label_encoders = {}
for col in ['subject_code_1', 'subject_code_2', 'subject_code_3','project_type', 'institution']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
data['money'] = scaler.fit_transform(data[['money']])

features = {
    'subject_code_1': torch.tensor(data['subject_code_1'].values),
    'subject_code_2': torch.tensor(data['subject_code_2'].values),
    'subject_code_3': torch.tensor(data['subject_code_3'].values),
    'project_type': torch.tensor(data['project_type'].values),
    'institution': torch.tensor(data['institution'].values),
    'money': torch.tensor(data['money'].values, dtype=torch.float)
}

embedding_dims = {
    # 'subject_code_1': 4,
    # 'subject_code_2': 4,
    # 'subject_code_3': 4,
    # 'project_type': 4,
    'institution': 4
}

embeddings = nn.ModuleDict({
    col: nn.Embedding(num_embeddings=len(label_encoders[col].classes_), 
                      embedding_dim=embedding_dim) 
    for col, embedding_dim in embedding_dims.items()
})

embedded_features = torch.cat([embeddings[col](features[col]) for col in embeddings], dim=1)

g = dgl.DGLGraph()
g.add_nodes(data.shape[0])
other_features = torch.stack([features[col] for col in ['subject_code_1', 'subject_code_2', 'subject_code_3', 'project_type', 'money']], dim=1)
all_features = torch.cat([embedded_features, other_features], dim=1)

g.ndata['features'] = all_features
g.ndata['label'] = torch.tensor(1-data['finished'].values)

institution_dict = {}
for i, inst_id in enumerate(data['institution']):
    if inst_id not in institution_dict:
        institution_dict[inst_id] = []
    institution_dict[inst_id].append(i)
person_dict = {}
for i, inst_id in enumerate(data['person']):
    if inst_id not in person_dict:
        person_dict[inst_id] = []
    person_dict[inst_id].append(i)

for inst_id, nodes in tqdm(person_dict.items()):
    src_nodes = []
    dst_nodes = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            src_nodes.append(nodes[i])
            dst_nodes.append(nodes[j])
            # 如果你的图是无向的，需要添加这行代码
            src_nodes.append(nodes[j])
            dst_nodes.append(nodes[i])
    total_edges = len(src_nodes)
    
    # num_edges_to_keep = int(total_edges * 0.3)
    
    # indices_to_keep = random.sample(range(total_edges), num_edges_to_keep)
    
    # src_nodes = [src_nodes[i] for i in indices_to_keep]
    # dst_nodes = [dst_nodes[i] for i in indices_to_keep]
    g.add_edges(src_nodes, dst_nodes)
num_label_0 = (g.ndata['label'] == 0).sum().item()
num_label_1 = (g.ndata['label'] == 1).sum().item()

num_to_delete = num_label_0 - int(num_label_1 / 0.005)
import numpy as np
nodes_to_delete = np.random.choice(np.where(g.ndata['label'] == 0)[0], size=num_to_delete, replace=False)

g.remove_nodes(nodes_to_delete)

# features_tensor = g.ndata['features']
# labels_tensor = g.ndata['label']
dgl.save_graphs('nsfc_per', g)
# dgl.save_graphs('nsfc_person', g)
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
            graph = dgl.load_graphs('/home/yuhanli/wangpeisong/GADBench/nsfc_per')[0][0]
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

for data_name in ['nsfc_per']:
    data = Dataset(data_name)
    data.split()
    print(data.graph)
    print(data.graph.ndata['train_masks'].sum(0), data.graph.ndata['val_masks'].sum(0), data.graph.ndata['test_masks'].sum(0))
    dgl.save_graphs(data_name, [data.graph])
