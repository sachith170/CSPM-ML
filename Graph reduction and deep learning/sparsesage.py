# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install torch torch-geometric
# !pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/posture_data.csv', sep=',')

!pip show torch
!pip show torch-geometric

# Commented out IPython magic to ensure Python compatibility.
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scale = StandardScaler()
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

X = TSNE(n_components=2, random_state=42).fit_transform( df )
X_scaled=scale.fit_transform(X)

kmeans_clustering = KMeans( n_clusters = 3 ).fit( X_scaled )
colors = np.array([x for x in 'cmyk'])
#plt.scatter(X_scaled[:,0], X_scaled[:,1], c=colors[kmeans_clustering.labels_], label='CSP 1')
for i in range(kmeans_clustering.n_clusters):
    X_scaled1 = X_scaled[kmeans_clustering.labels_ == i]
    plt.scatter(X_scaled1[:, 0], X_scaled1[:, 1], c=colors[i], label=f'CSP {i + 1}')

plt.legend()
#plt.title('K-Means Clusters')
plt.savefig("K-Means.png")
plt.show()

import dgl
import dgl.nn as dglnn
from dgl.nn import SAGEConv
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, roc_curve, f1_score, average_precision_score
import random
import networkx as nx
import time
from torch.utils.data.sampler import SubsetRandomSampler
import gc

#import torch
import torch_geometric
from typing import Tuple
from tqdm import tqdm

class EfficientOneWalkIndexSparsifier:
    """
    Efficient implementation for the 1-Walk Index Sparsification (WIS) algorithm based only on vertex degrees (see Algorithm 2 in the paper).
    """

    def __remove_self_loops(self, remaining_edges, curr_indices):
        mask = remaining_edges[0] != remaining_edges[1]
        remaining_edges = remaining_edges[:, mask]
        curr_indices = curr_indices[mask]
        return remaining_edges, curr_indices

    def __compute_index_of_edge_to_remove(self, remaining_edges, vertex_degrees):
        per_edge_degrees = vertex_degrees[remaining_edges]
        larger_degree_side_per_edge = torch.max(per_edge_degrees, dim=0)[0]
        smaller_degree_side_per_edge = torch.min(per_edge_degrees, dim=0)[0]

        max_min_degree = torch.max(smaller_degree_side_per_edge)
        maximal_min_degree_edges = smaller_degree_side_per_edge == max_min_degree
        maximal_max_degree_edge = torch.argmax(larger_degree_side_per_edge[maximal_min_degree_edges])

        index_of_edge_to_remove = torch.where(maximal_min_degree_edges)[0][maximal_max_degree_edge]
        return index_of_edge_to_remove

    def sparsify(self, num_vertices: int, edge_index: torch.Tensor, num_edges_to_remove: int, undirected: bool = False,
                 print_progress: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes edge_index of sparsified graph for the 1-WIS algorithm (efficient implementation).
        @param num_vertices: Number of vertices in the graph.
        @param edge_index: Tensor of shape (2, num_edges) describing the graph edges according to PyTorch Geometric format (first row is source and
        second row is target).
        @param num_edges_to_remove: Number of edges to remove.
        @param undirected: Whether to treat edges as undirected or not.
        @param device: PyTorch device to use.
        @param print_progress: Whether to print iteration of edge removal progress.
        @return: A tuple consisting of: (i) A tensor of shape (2, num_edges_remaining) containing the remaining edges;
        (ii) a tensor of shape (2, num_edges_to_remove) containing the removed edges in order of removal (first to last), where for undirected graphs only one side per edge appears;
        and (iii) a tensor holding the indices of all removed edges in the original edge_index in order of removal (first to last), where for undirected graphs the indices of both
        directions of a removed edge appear.
        """
        #remaining_edges = edge_index
        remaining_edges=torch.stack(edge_index, dim=0)
        #print('remain',remaining_edges)
        curr_indices = torch.arange(0, remaining_edges.shape[1])

        # Remove self-loops, as they are not removed by the algorithm
        remaining_edges, curr_indices = self.__remove_self_loops(remaining_edges, curr_indices)

        edges_to_remove_by_order = [[], []]
        indices_of_removed_edges = []
        vertex_degrees = torch_geometric.utils.degree(remaining_edges[1], num_nodes=num_vertices)

        for _ in tqdm(range(num_edges_to_remove), disable=not print_progress):
            if remaining_edges.shape[1] == 0:
                break

            index_of_edge_to_remove = self.__compute_index_of_edge_to_remove(remaining_edges, vertex_degrees)
            edges_to_remove_by_order[0].append(remaining_edges[0][index_of_edge_to_remove].item())
            edges_to_remove_by_order[1].append(remaining_edges[1][index_of_edge_to_remove].item())

            vertex_degrees[remaining_edges[1][index_of_edge_to_remove]] -= 1

            edges_to_remove = remaining_edges[:, index_of_edge_to_remove].unsqueeze(dim=1)
            if undirected:
                vertex_degrees[remaining_edges[0][index_of_edge_to_remove]] -= 1
                other_dir_edge_to_remove = torch.stack([edges_to_remove[1], edges_to_remove[0]])
                edges_to_remove = torch.concat([edges_to_remove, other_dir_edge_to_remove], dim=1)

            mask_to_remove = torch.any(torch.all(remaining_edges[:, None, :] == edges_to_remove[:, :, None], dim=0), dim=0)

            indices_of_removed_edges.append(curr_indices[mask_to_remove])
            curr_indices = curr_indices[~mask_to_remove]
            remaining_edges = remaining_edges[:, ~mask_to_remove]

        edges_to_remove_by_order = torch.tensor(edges_to_remove_by_order)
        indices_of_removed_edges = torch.cat(indices_of_removed_edges)
        return remaining_edges, edges_to_remove_by_order, indices_of_removed_edges

object_columns = df.select_dtypes(include=[object]).columns
df[object_columns] = df[object_columns].apply(pd.to_numeric, errors='coerce')
features = torch.tensor(df.iloc[:].values, dtype=torch.float)
#print(features)
labels = torch.tensor(kmeans_clustering.labels_, dtype=torch.long)

graph1 = dgl.graph(([], []))

num_nodes = len(df)
graph1.add_nodes(num_nodes)

if num_nodes > 1:

    for i in range(num_nodes):
        graph1.add_edges(i, (i + 1) % num_nodes)
        graph1.add_edges((i + 1) % num_nodes, i)

def add_random_edge(graph1, num_nodes):
    source = random.randint(0, num_nodes - 1)
    target = random.randint(0, num_nodes - 1)
    if source != target:
        if random.choice([True, False]):
            graph1.add_edges(source, target)
        else:
            #graph.add_edges(source, target)
            graph1.add_edges(target, source)

num_interconnecting_edges = 500

for _ in range(num_interconnecting_edges):
    add_random_edge(graph1, num_nodes)

graph1.ndata['features'] = features
graph1.ndata['labels'] = labels

import pickle
with open('/content/drive/MyDrive/graph1.pkl', 'wb') as f:
    pickle.dump(graph1, f)

import pickle
with open('/content/drive/MyDrive/graph.pkl', 'rb') as file:
    graph = pickle.load(file)

features = graph.ndata['features']
labels = graph.ndata['labels']

import gc
gc.collect()

gc.collect()

num_nodes = len(df)
input_size = features.size(1)
hidden_size = 64
output_size = len(torch.unique(labels))

class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, num_classes, 'mean')

    def forward(self, sparsified_graph, features):
        x = self.conv1(sparsified_graph, features)
        x = F.relu(x)
        x = self.conv2(sparsified_graph, x)
        return x

num_train_nodes = int(0.7 * len(df))
train_idx = list(range(num_train_nodes))

num_test_nodes =  num_nodes - num_train_nodes
test_idx = list(range(num_test_nodes))

batch_size = 100
train_loader = GraphDataLoader(list(train_idx), batch_size=batch_size, shuffle=True)
test_loader = GraphDataLoader(list(test_idx), batch_size=batch_size, shuffle=True)

#print("Features shape:", features.shape)
#print("Graph nodes:", graph.number_of_nodes())
print("Graph before removal edges:", graph.number_of_edges())
print(graph.edges())
print(type(graph.edges()))
edge_index=torch.stack(graph.edges(), dim=0)
#edge_index=torch.tensor(graph.edges())
print(edge_index)
print(type(edge_index.shape))
#edge_index=torch.tensor(graph.number_of_edges(), dtype=torch.int8)
#print(graph.edges())
num_undirected_edges = edge_index.shape[1]

import gc
gc.collect()

unique_labels = torch.unique(labels)

model = GNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

def train():
    model.train()
    #start_time = time.time()
    total_loss = 0
    for batched_train_idx in train_loader:
        optimizer.zero_grad()
        batched_graph = dgl.node_subgraph(sparsified_graph, batched_train_idx)
        logits = model(batched_graph, features[batched_train_idx])
        loss = F.cross_entropy(logits, labels[batched_train_idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test():
    model.eval()
    test_predictions = []
    predicted_probs =[]
    test_labels = []
    testing_loss = 0.0
    batch_indices = []
    with torch.no_grad():
        for batched_test_idx in test_loader:
            batch_indices.append(batched_test_idx)
            #global test_graph
            test_graph = dgl.node_subgraph(sparsified_graph, batched_test_idx)
            logits = model(test_graph, features[batched_test_idx])
            predicted_labels = torch.argmax(logits, dim=1).numpy()
            predicted_prob = F.softmax(logits, dim=1).numpy()
            test_predictions.extend(predicted_labels)
            predicted_probs.extend(predicted_prob)
            test_labels.extend(labels[batched_test_idx].numpy())
            test_labels_tensor = torch.tensor(predicted_labels, dtype=torch.long)
            testing_loss += F.cross_entropy(logits, test_labels_tensor).item()

    accuracy = accuracy_score(test_labels, test_predictions)
    testing_loss /= len(test_loader)

    return accuracy, testing_loss, test_predictions, batch_indices,predicted_probs

num_epochs = 4000

model = GNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

losses = []
accuracies = []
training_times = []
testing_losses = []
testing_predictions = []

number_edges_to_remove = 500  # edges to remove at each iteration

iteration_count=0
previous_accuracy=0
new_accuracy=0

while True:
    one_wis = EfficientOneWalkIndexSparsifier()
    #if iteration_count==0:
     # iteration_count+=1
    remaining_edges, edges_removed, indices_of_removed_edges = one_wis.sparsify(graph.number_of_nodes(), graph.edges(),
                                                                            num_edges_to_remove=number_edges_to_remove, print_progress=True, undirected=False)
    print(f"Number directed edges after removal: {remaining_edges.shape[1]}")

    sparsified_graph = dgl.graph(tuple(remaining_edges.tolist()), num_nodes=num_nodes)
    number_edges_to_remove+=500
    #else:
     # remaining_edges, edges_removed, indices_of_removed_edges = one_wis.sparsify(graph.number_of_nodes(), graph.edges(),
     #                                                                       num_edges_to_remove=number_edges_to_remove, print_progress=True, undirected=False)
      #print(f"Number directed edges after removal: {remaining_edges.shape[1]}")

      #sparsified_graph = dgl.graph(tuple(remaining_edges.tolist()), num_nodes=num_nodes)
      #number_edges_to_remove+=1000
    for epoch in range(4000):
      train()
      accuracy, testing_loss, test_predictions, batch_indices, predicted_probs = test()

    print(f'New Test accuracy: {accuracy:.4f}')
    del sparsified_graph
    #if previous_accuracy==0:
     # previous_accuracy=accuracy
    #  new_accuracy=accuracy
    #else:
     # new_accuracy=accuracy
    #del sparsified_graph
    #gc.collect()
    if accuracy >= 0.9:
        print('Maintained accuracy, continuing sparsification.')
        del accuracy, testing_loss, test_predictions, batch_indices, predicted_probs
    else:
        print('Accuracy dropped below threshold')
        break

sparsified_graph.number_of_edges()

import matplotlib.pyplot as plt
for epoch in range(num_epochs):
    start_time = time.time()
    loss = train()
    end_time = time.time()
    epoch_time = end_time - start_time
    training_times.append(epoch_time)
    accuracy, testing_loss, test_predictions, batch_indices, predicted_probs = test()
    losses.append(loss)
    #training_times.append(training_time)
    accuracies.append(accuracy)
    testing_losses.append(testing_loss)
    testing_predictions.append(test_predictions)

plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, 3.25])
plt.savefig("Training Loss over Epochs sparse.png")
plt.show()

average_epoch_time = sum(training_times) / num_epochs
print(f"Average Training Time per Epoch: {average_epoch_time:.2f} seconds")

plt.plot(range(1, num_epochs + 1), testing_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.1, 0.6])
plt.savefig("Test Loss over Epochs sparse.png")
plt.show()

plt.plot(range(1, num_epochs + 1), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("test Accuracy over Epochs sparse.png")
plt.show()

batch_indices_tensor = torch.cat(batch_indices)
test_predictions_np = np.array(test_predictions)
predicted_probs_np=np.array(predicted_probs)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(output_size):
    class_labels = (labels[batch_indices_tensor] == i).numpy().astype(int)
    precision[i], recall[i], _ = precision_recall_curve(class_labels, predicted_probs_np[:, i])
    average_precision[i] = average_precision_score(class_labels, predicted_probs_np[:, i])

all_precision = np.unique(np.concatenate([precision[i] for i in range(output_size)]))
mean_recall = np.zeros_like(all_precision)
for i in range(output_size):
    mean_recall += np.interp(all_precision, precision[i], recall[i])
mean_recall /= output_size
overall_average_precision = np.mean([average_precision[i] for i in range(output_size)])

plt.plot(mean_recall, all_precision, lw=2, label='PR curve (AP = {0:0.2f})'.format(overall_average_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.savefig("PR sparse.png")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(labels[batch_indices_tensor], test_predictions_np)
class_percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

disp = ConfusionMatrixDisplay(confusion_matrix=class_percentages, display_labels=["CSP 1", "CSP 2", "CSP 3"])

disp.plot(cmap=plt.cm.Blues, values_format='.2%')

plt.savefig("Confusion Matrix sparse.png")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(labels[batch_indices_tensor], test_predictions_np)
class_percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["CSP 1", "CSP 2", "CSP 3"])

disp.plot(cmap=plt.cm.Blues)

plt.savefig("Confusion Matrix values sparse.png")
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(output_size):
    class_labels = (labels[batch_indices_tensor] == i).numpy().astype(int)
    fpr[i], tpr[i], _ = roc_curve(class_labels, predicted_probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(output_size)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(output_size):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= output_size
overall_roc_auc = auc(all_fpr, mean_tpr)

plt.plot(all_fpr, mean_tpr, lw=2, label='ROC curve (AUC = {0:0.2f})'.format(overall_roc_auc))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("ROC sparse.png")
plt.show()

train_f1_scores = []
valid_f1_scores = []
train_nosparse_f1_scores = []
valid_nosparse_f1_scores = []

def train():
    model.train()
    total_loss = 0
    for batched_train_idx in train_loader:
        optimizer.zero_grad()
        batched_graph = dgl.node_subgraph(sparsified_graph, batched_train_idx)
        logits = model(batched_graph, features[batched_train_idx])
        predicted_labels = torch.argmax(logits, dim=1).numpy()
        loss = F.cross_entropy(logits, labels[batched_train_idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_f1 = f1_score(labels[batched_train_idx], predicted_labels, average='micro')
    return total_loss / len(train_loader), train_f1

def test():
    model.eval()
    with torch.no_grad():
      for batched_test_idx in test_loader:
        test_graph = dgl.node_subgraph(sparsified_graph, batched_test_idx)
        logits = model(test_graph, features[batched_test_idx])
        predicted_labels = torch.argmax(logits, dim=1).numpy()
    valid_f1 = f1_score(labels[batched_test_idx], predicted_labels, average='micro')
    return valid_f1

def nosparse_train():
    model.train()
    nosparse_total_loss = 0
    for batched_train_idx in train_loader:
        optimizer.zero_grad()
        batched_graph = dgl.node_subgraph(graph, batched_train_idx)
        logits = model(batched_graph, features[batched_train_idx])
        predicted_labels = torch.argmax(logits, dim=1).numpy()
        loss = F.cross_entropy(logits, labels[batched_train_idx])
        loss.backward()
        optimizer.step()
        nosparse_total_loss += loss.item()
    nosparse_train_f1 = f1_score(labels[batched_train_idx], predicted_labels, average='micro')
    return nosparse_total_loss / len(train_loader), nosparse_train_f1

def nosparse_test():
    model.eval()
    with torch.no_grad():
      for batched_test_idx in test_loader:
        test_graph = dgl.node_subgraph(graph, batched_test_idx)
        logits = model(test_graph, features[batched_test_idx])
        predicted_labels = torch.argmax(logits, dim=1).numpy()
    nosparse_valid_f1 = f1_score(labels[batched_test_idx], predicted_labels, average='micro')
    return nosparse_valid_f1

for epoch in range(499, 4001, 500):
    loss, train_f1 = train()
    valid_f1 = test()

    loss, nosparse_train_f1 = nosparse_train()
    nosparse_valid_f1 = nosparse_test()

    train_f1_scores.append(train_f1)
    valid_f1_scores.append(valid_f1)

    train_nosparse_f1_scores.append(nosparse_train_f1)
    valid_nosparse_f1_scores.append(nosparse_valid_f1)

    plt.bar(epoch + 1, valid_f1, color='blue',align='edge', width = 40, label='With sparsification' if (epoch+1) == 500 else "")
    plt.bar(epoch + 1, nosparse_valid_f1, color='orange',align='edge', width = -40, label='Without sparsification' if (epoch+1) == 500 else "")

plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend(loc='center right')
plt.savefig("F1-Score Over Epochs.png")
plt.show()
