import torch
import networkx as nx
import numpy as np
from scipy.special import beta
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import combinations
from torch_geometric.loader import DataLoader
from collections import defaultdict
import time


# Parameter Settings

k = 2
d = 64
y = 0.5
torch.manual_seed(42)


# Datasets Selection

data = pd.read_excel("Datasets/39AIDS_edge.xlsx")#39AIDS,54BZR,54DHFR,4ENZYMES,36COX2
node = pd.read_excel("Datasets/39AIDS_node.xlsx")
n = 39
Dim = 64
lr = 0.005

# data = pd.read_excel("54BZR_edge.xlsx")
# node = pd.read_excel("54BZR_node.xlsx")
# n = 54
# Dim = 64
# lr = 0.007

# data = pd.read_excel("54DHFR_edge.xlsx")
# node = pd.read_excel("54DHFR_node.xlsx")
# n = 54
# Dim = 64
# lr = 0.006

# data = pd.read_excel("4ENZYMES_edge.xlsx")
# node = pd.read_excel("4ENZYMES_node.xlsx")
# n = 4
# Dim = 64
# lr = 0.007

# data = pd.read_excel("36COX2_edge.xlsx")
# node = pd.read_excel("36COX2_node.xlsx")
# n = 36
# Dim = 64
# lr = 0.007



# Dataset Processing

graphs = []
w = []

node_to_graph = {}

for i, row in node.iterrows():

    node_id = int(i + 1)

    node_to_graph[node_id] = row["graph_idx"]


graph_edges = defaultdict(list)

for row in tqdm(
    data.itertuples(index=False),
    total=len(data),
    desc="Grouping edges"
):

    source = int(row.source)
    target = int(row.target)


    source_graph = node_to_graph.get(source)
    target_graph = node_to_graph.get(target)

    if (
        source_graph is not None
        and target_graph is not None
        and source_graph == target_graph
    ):

        graph_edges[source_graph].append(
            (source, target)
        )


graphs = []

for graph_id, node_graph in tqdm(
    node.groupby("graph_idx"),
    total=node["graph_idx"].nunique(),
    desc="Building graphs"
):

    G = nx.Graph()

    for i, row in node_graph.iterrows():

        feature = [
            float(x)
            for x in str(row["node_attrs"]).split(",")
        ]

        G.add_node(
            int(i + 1),
            label=int(row["node_label"]),
            feature=feature
        )

    for source, target in graph_edges.get(graph_id, []):

        if source in G.nodes and target in G.nodes:

            G.add_edge(
                source,
                target
            )

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

            similarity = (
                len(neighbors_a & neighbors_b)
                / len(union)
            )

        weight = 1.0 - similarity

        G[a][b]["weight"] = weight

        w.append(weight)

    if len(w) > 0:

        min_w = min(w)
        max_w = max(w)

    else:

        min_w = 0.0
        max_w = 0.0


    for a, b in G.edges():

        weight = G[a][b]["weight"]

        if max_w == min_w:

            G[a][b]["weight"] = 0.0

        else:

            G[a][b]["weight"] = (
                (weight - min_w)
                / (max_w - min_w)
            )


    graphs.append(G)


print("========================================")
print("Graph construction finished")
print("Number of graphs:", len(graphs))
print("========================================")


train_graphs,test_graphs=train_test_split(graphs,test_size=0.2,random_state = 42)
train_graphs, val_graphs = train_test_split(train_graphs,test_size=0.25,random_state=42)

print(len(train_graphs))
print(len(test_graphs))
print(len(val_graphs))


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
    F_k_i = np.array(F_k_i, )
    return F_k_i, max_comb


# Low-Order Encoding Module

def node(G):
    nodes = list(G.nodes())

    node_feature = [
        G.nodes[i]["feature"]
        for i in nodes
    ]

    x = torch.tensor(
        node_feature,
        dtype=torch.float
    )

    '''degree = G.degree()
    degree_feature = [degree[node] for node in nodes]
    node_feature = np.array(degree_feature).reshape(-1,1)
    x = torch.tensor(node_feature,dtype=torch.float)'''

    edge = []
    edge_weight = []
    for a,b,data in G.edges(data=True):
        i = nodes.index(a)
        j = nodes.index(b)
        edge.append([i,j])
        edge.append([j,i])
        edge_weight.append(data['weight'])
        edge_weight.append(data['weight'])
    edge = torch.tensor(edge,dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weight,dtype=torch.float)
    y = [G.nodes[node]['label'] for node in nodes]
    y = torch.tensor(y,dtype=torch.long)
    data = Data(x = x,edge_index = edge,edge_weight = edge_weight,y = y)
    return data


class GCN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            64
        )

        self.conv2 = GCNConv(
            64,
            d
        )

    def forward(self, data):

        x = data.x

        edge_index = data.edge_index

        edge_weight = data.edge_weight

        x = self.conv1(
            x,
            edge_index,
            edge_weight
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x

sample_data = node(graphs[0])

gcn = GCN(sample_data.num_features)



# Multi-Order Feature Rectification

class AttentionMLP(nn.Module):

    def __init__(self, gcn_dim, pp_dim, num_classes):

        super().__init__()

        self.gcn_proj = nn.Linear(
            gcn_dim,
            d
        )
        # Learnable Persistence Probability
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

        output = self.mlp(H)

        return output

model = AttentionMLP(d,Dim,n)

# Dataset Splitting

train_data = []

print("\n==============================")
print("计算训练集 PP 特征")
print("==============================")
pp_start = time.perf_counter()
for G in tqdm(train_graphs):

    data_train = node(G)

    F, max_comb = PP(
        G,
        y,
        k,
        Dim
    )

    H_PP = np.array(F)

    H_PP = torch.tensor(
        H_PP,
        dtype=torch.float
    )

    data_train.H_PP = H_PP

    train_data.append(data_train)


val_data = []

print("\n==============================")
print("计算验证集 PP 特征")
print("==============================")

for G in tqdm(val_graphs):

    data_val = node(G)

    F, max_comb = PP(
        G,
        y,
        k,
        Dim
    )

    H_PP = np.array(F)

    H_PP = torch.tensor(
        H_PP,
        dtype=torch.float
    )

    data_val.H_PP = H_PP

    val_data.append(data_val)

test_data = []

print("\n==============================")
print("计算测试集 PP 特征")
print("==============================")

for G in tqdm(test_graphs):

    data_test = node(G)

    F, max_comb = PP(
        G,
        y,
        k,
        Dim
    )

    H_PP = np.array(F)

    H_PP = torch.tensor(
        H_PP,
        dtype=torch.float
    )

    data_test.H_PP = H_PP

    test_data.append(data_test)
pp_end = time.perf_counter()

pp_time = pp_end - pp_start
print("PP计算时间: {:.4f}s".format(pp_time))

train_loader = DataLoader(
    train_data,
    batch_size=len(train_data),
    shuffle = False
)

val_loader = DataLoader(
    val_data,
    batch_size=len(val_data),
    shuffle = False
)

test_loader = DataLoader(
    test_data,
    batch_size=len(test_data),
    shuffle = False
)

# Training Module

optimizer = torch.optim.Adam(
    list(gcn.parameters()) +
    list(model.parameters()),
    lr=lr,
    weight_decay=5e-4
)

criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

best_epoch = 0

best_gcn_state = None

best_model_state = None
train_start = time.perf_counter()
for epoch in range(1000):

    gcn.train()
    model.train()

    total_loss = 0.0

    train_correct = 0
    train_total = 0

    for batch in train_loader:

        H_GCN = gcn(batch)

        H_PP = batch.H_PP

        output = model(
            H_GCN,
            H_PP
        )

        loss = criterion(
            output,
            batch.y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        prediction = torch.argmax(
            output,
            dim=1
        )

        train_correct += (
            prediction == batch.y
        ).sum().item()

        train_total += len(
            batch.y
        )


    train_acc = (
        train_correct /
        train_total
    )

    gcn.eval()
    model.eval()

    val_correct = 0
    val_total = 0

    val_loss = 0.0


    with torch.no_grad():

        for batch in val_loader:

            H_GCN = gcn(batch)

            H_PP = batch.H_PP

            output = model(
                H_GCN,
                H_PP
            )

            loss = criterion(
                output,
                batch.y
            )

            val_loss += loss.item()

            prediction = torch.argmax(
                output,
                dim=1
            )

            val_correct += (
                prediction == batch.y
            ).sum().item()

            val_total += len(
                batch.y
            )


    val_acc = (
        val_correct /
        val_total
    )

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
            total_loss / len(train_loader),
            "train_acc:",
            train_acc,
            "val_loss:",
            val_loss / len(val_loader),
            "val_acc:",
            val_acc
        )

gcn.load_state_dict(
    best_gcn_state
)

model.load_state_dict(
    best_model_state
)

train_end = time.perf_counter()

train_time = train_end - train_start
print("模型训练时间: {:.4f}s".format(train_time))


print()
print("==============================")
print("最佳模型")
print("==============================")

print(
    "Best Epoch:",
    best_epoch
)

print(
    "Best Validation Accuracy:",
    best_val_acc
)

# Prediction Module

gcn.eval()
model.eval()
inference_start = time.perf_counter()
test_correct = 0
test_total = 0

test_loss = 0.0


with torch.no_grad():

    for batch in test_loader:


        H_GCN = gcn(batch)

        H_PP = batch.H_PP

        output = model(
            H_GCN,
            H_PP
        )

        loss = criterion(
            output,
            batch.y
        )

        test_loss += loss.item()

        prediction = torch.argmax(
            output,
            dim=1
        )

        test_correct += (
            prediction == batch.y
        ).sum().item()

        test_total += len(
            batch.y
        )
inference_end = time.perf_counter()

inference_time = inference_end - inference_start

print("推理时间: {:.6f}s".format(inference_time))
test_acc = (
    test_correct /
    test_total
)


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
    train_acc
)

print(
    "Validation Accuracy:",
    best_val_acc
)

print(
    "Test Accuracy:",
    test_acc
)

print(
    "Test Loss:",
    test_loss / len(test_loader)
)

print(
    "Test Correct:",
    test_correct
)

print(
    "Test Total:",
    test_total
)

print("========================================")