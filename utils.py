from karateclub import NetMF
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import pandas as pd
# NetMF embedding logic using karate club module
def netmf_embedding(graph, dimensions=16, order=2):
    model = NetMF(dimensions=dimensions, order=order,seed=21)
    model.fit(graph)
    return model.get_embedding()

def create_data_object(sub_conn_matrix, label, dimensions, order, threshold=0.5):
    adjacency_matrix = (sub_conn_matrix > threshold).astype(float)
    graph = nx.from_numpy_matrix(adjacency_matrix)
    netmf_embeddings = netmf_embedding(graph,dimensions, order)  # Replace this with your actual embedding logic
    
    # Extract node features from the embeddings
    node_features = torch.tensor(netmf_embeddings, dtype=torch.float32)
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(label,dtype=torch.long))
    
    return data

def create_labels():
    label=[]
    subject_fc_matrices=np.load('source_data/fc/fc_matrices.npy')
    print(f"Number of Subjects {len(subject_fc_matrices)}")
    for i in range(len(subject_fc_matrices)):
        if i <25:
            label.append(0)
        if i >= 25 and i<48:
            label.append(1)
        if i >=48:
            label.append(2)

    all_labels = np.array(label)
    num_classes=len(set(label))
    # Find unique labels and their counts
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    # Create a dictionary to store the counts for each unique label
    label_counts = dict(zip(unique_labels, counts))
    # Print the results
    for label, count in label_counts.items():
        print(f"Label {label}: {count} subjects")
    return subject_fc_matrices,all_labels,num_classes