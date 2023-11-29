import dgl
import torch
path='your path to dataset'
g = dgl.load_graphs(path)
graph=g[0][0]
num_nodes = graph.number_of_nodes()

num_edges = graph.number_of_edges()//2

feature_dim = graph.ndata['features'].shape[1]

labels = graph.ndata['label']
zero_label_ratio = torch.sum(labels == 0).item() / len(labels)

print(f"#Node: {num_nodes}")
print(f"#Edge: {num_edges}")
print(f"#Feature: {feature_dim}")
print(f"#Ratio of 0: {zero_label_ratio}")