import torch
import networkx as nx
import numpy as np
from scipy.special import beta
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import combinations
import time
from torch_geometric.datasets import Coauthor,Planetoid,Amazon

# Parameter Settings

k = 2
d = 64
y = 0.5
torch.manual_seed(42)

# Datasets Selection

dataset = Planetoid(root='./data',name='Cora')
Dim = 128
lr = 0.008

# dataset = Amazon(root='./data',name='Photo')
# Dim = 128
# lr = 0.005

# dataset = Amazon(root='./data',name='Computers')
# Dim = 64
# lr = 0.005

# dataset = Coauthor(root='./data',name='CS')
# Dim = 64
# lr = 0.009

# dataset = Coauthor(root='./data',name='Physics')
# Dim = 64
# lr = 0.01


# Dataset Splitting

data = dataset[0]

num_nodes = data.num_nodes

idx = np.arange(num_nodes)

train_idx, test_idx = train_test_split(
    idx,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.25,
    random_state=42
)

train_mask = torch.zeros(num_nodes,dtype=torch.bool)

val_mask = torch.zeros(num_nodes,dtype=torch.bool)

test_mask = torch.zeros(num_nodes,dtype=torch.bool)


train_mask[train_idx]=True
val_mask[val_idx]=True
test_mask[test_idx]=True


# Dataset Processing

G = nx.Graph()

for i in range(data.num_nodes):
    G.add_node(i, label=int(data.y[i]))

for i in range(data.edge_index.shape[1]):
    a = int(data.edge_index[0, i])
    b = int(data.edge_index[1, i])
    G.add_edge(a, b)


# Edge Weight Partitioning

w = []

for a, b in G.edges():

    neighbors_a = set(G.neighbors(a))
    neighbors_b = set(G.neighbors(b))

    neighbors_a.discard(b)
    neighbors_b.discard(a)

    union = neighbors_a | neighbors_b

    if len(union) == 0:
        similarity = 0.0
    else:
        similarity = len(neighbors_a & neighbors_b) / len(union)

    weight = 1.0 - similarity

    G[a][b]['weight'] = weight
    w.append(weight)

min_w = min(w)
max_w = max(w)

for a, b in G.edges():

    weight = G[a][b]['weight']

    if max_w == min_w:
        G[a][b]['weight'] = 0.0
    else:
        G[a][b]['weight'] = (
            (weight - min_w) / (max_w - min_w)
        )

max_degree = max(dict(G.degree()).values())
n = dataset.num_classes


# Higher-Order Encoding Module

def PP(G,y,k,Dim):

    F_k_i = []
    max_comb = 0
    for xi in tqdm(G.nodes()):
        neighbors = []
        for j in G.neighbors(xi):
            if G[xi][j]['weight'] <= y:
                neighbors.append(j)
        if len(neighbors) >= k:
            num_comb = len(list(combinations(neighbors, k)))
            max_comb = max(max_comb, num_comb)
    for xi in tqdm(G.nodes()):
        neighbors = []

        for j in G.neighbors(xi):
            if G[xi][j]['weight'] <= y:
                neighbors.append(j)

        if len(neighbors) < k :
            F_k_i.append([0] * Dim)
            continue

        if len(neighbors) >= k:
            num_comb = len(list(combinations(neighbors, k)))
            max_comb = max(max_comb, num_comb)

        F_k = []
        for nodes_k in combinations(neighbors, k):
            node = list(nodes_k)
            node.append(xi)
            b = []

            for i in node:
                time = []
                nodes = node.copy()
                nodes.remove(i)
                for u in nodes:
                    weights = G[i][u]['weight'] if G.has_edge(i,u) else 0
                    time.append(1-weights)

                b.append(min(time))

            s = np.sum(b)
            if s == 0:
                f = 0
            else:
                B = beta(s, k)
                p = np.prod(b)
                f = (k * (k + 1) * B * p)
            F_k.append(float(f))
        while len(F_k) < max_comb:
            F_k.append(0)
        F_k.sort(reverse=True)
        if len(F_k) >= Dim:
            F_k = F_k[:Dim]
        else:
            F_k.extend(
                [0.0] * (Dim - len(F_k))
            )
        F_k_i.append(F_k)
    F_k_i = np.array(F_k_i)
    return F_k_i, max_comb

pp_start = time.perf_counter()

F, max_comb = PP(G, y, k, Dim)

pp_end = time.perf_counter()

pp_time = pp_end - pp_start

print("PP计算时间: {:.4f} 秒".format(pp_time))
H = np.array(F)
H_PP = torch.tensor(H,dtype=torch.float)

# Low-Order Encoding Module

def node(G,data):
    nodes = list(G.nodes())
    """
    degree = G.degree()
    degree_feature = [degree[node] for node in nodes]
    node_feature = np.array(degree_feature).reshape(-1,1)
    x = torch.tensor(node_feature,dtype=torch.float)
    """
    x = data.x
    edge = []
    edge_weight = []
    for a,b,d in G.edges(data=True):
        i = nodes.index(a)
        j = nodes.index(b)
        edge.append([i,j])
        edge.append([j,i])
        edge_weight.append(d['weight'])
        edge_weight.append(d['weight'])
    edge = torch.tensor(edge,dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weight,dtype=torch.float)
    y = [G.nodes[node]['label'] for node in nodes]
    y = torch.tensor(y,dtype=torch.long)
    data1 = Data(x = x,edge_index = edge,edge_weight = edge_weight,y = y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
                 )
    return data1

data1 = node(G,data)


class GCN(nn.Module):

    def __init__(self,):

        super().__init__()


        self.conv1=GCNConv(
            data.num_features,
            64
        )


        self.conv2=GCNConv(
            64,
            d
        )


    def forward(self,data):

        x=data.x
        edge_index=data.edge_index
        edge_weight = data1.edge_weight

        x=self.conv1(
            x,
            edge_index,
            edge_weight
        )

        x=torch.relu(x)

        x=self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x

gcn = GCN()


# Multi-Order Feature Rectification

class AttentionMLP(nn.Module):

    def __init__(self, gcn_dim, pp_dim, num_classes):

        super().__init__()

        self.gcn_proj = nn.Linear(
            gcn_dim,
            d
        )

        self.pp_proj = nn.Linear(
            pp_dim,
            d
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=4,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, num_classes)
        )

    def forward(self, H_GCN, H_PP):

        gcn = self.gcn_proj(H_GCN)

        pp = self.pp_proj(H_PP)

        H = torch.stack(
            [gcn, pp],
            dim=1
        )

        # Attention
        H, att = self.attention(
            H,
            H,
            H
        )

        H = H.mean(dim=1)

        # MLP
        output = self.mlp(H)

        return output

model = AttentionMLP(d,Dim,dataset.num_classes)


# Training Module

optimizer = torch.optim.Adam(
    list(gcn.parameters()) +
    list(model.parameters()),
    lr,
    weight_decay=5e-4
)

criterion = nn.CrossEntropyLoss()
train_start = time.perf_counter()
best_val_acc = 0.0
best_epoch = 0
best_model_state = None
best_gcn_state = None

for epoch in range(1000):

    gcn.train()
    model.train()

    H_GCN = gcn(data1)

    output = model(
        H_GCN,
        H_PP
    )

    loss = criterion(
        output[data1.train_mask],
        data1.y[data1.train_mask]
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    gcn.eval()
    model.eval()

    with torch.no_grad():

        H_GCN = gcn(data1)

        val_output = model(
            H_GCN,
            H_PP
        )

        val_pred = torch.argmax(
            val_output,
            dim=1
        )

        train_acc = (
            val_pred[data1.train_mask]
            == data1.y[data1.train_mask]
        ).float().mean().item()

        val_acc = (
            val_pred[data1.val_mask]
            == data1.y[data1.val_mask]
        ).float().mean().item()

        test_acc = (
            val_pred[data1.test_mask]
            == data1.y[data1.test_mask]
        ).float().mean().item()

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_epoch = epoch

        best_gcn_state = {
            key: value.detach().cpu().clone()
            for key, value in gcn.state_dict().items()
        }

        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    if epoch % 10 == 0:

        print(
            "epoch:",
            epoch,
            "loss:",
            loss.item(),
            "train_acc:",
            train_acc,
            "val_acc:",
            val_acc,
            "test_acc:",
            test_acc
        )

gcn.load_state_dict(
    best_gcn_state
)

model.load_state_dict(
    best_model_state
)
train_end = time.perf_counter()

train_time = train_end - train_start

print("模型训练时间: {:.4f} 秒".format(train_time))
print("模型训练时间: {:.4f} 分钟".format(train_time / 60))

torch.save(
    gcn.state_dict(),
    "./PP node classification/gcn.pth"
)


# Prediction Module

gcn.eval()
model.eval()

inference_start = time.perf_counter()

with torch.no_grad():

    H_GCN = gcn(data1)

    output = model(
        H_GCN,
        H_PP
    )

    prediction = torch.argmax(
        output,
        dim=1
    )

inference_end = time.perf_counter()

inference_time = inference_end - inference_start

print("推理时间: {:.6f} 秒".format(inference_time))
final_train_acc = (
    prediction[data1.train_mask]
    == data1.y[data1.train_mask]
).float().mean().item()

final_val_acc = (
    prediction[data1.val_mask]
    == data1.y[data1.val_mask]
).float().mean().item()

final_test_acc = (
    prediction[data1.test_mask]
    == data1.y[data1.test_mask]
).float().mean().item()

print()
print()
print("========================================")
print("             OUR MODEL RESULT")
print("========================================")

print(
    "Best Epoch:",
    best_epoch
)

print(
    "Train Accuracy:",
    final_train_acc
)

print(
    "Validation Accuracy:",
    final_val_acc
)

print(
    "Test Accuracy:",
    final_test_acc
)

print("========================================")