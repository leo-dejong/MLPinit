from node2vec import Node2Vec
import networkx as nx
from torch_geometric.utils import to_networkx

def generate_node2vec_embeddings(data, dimensions=2, walk_length=2, num_walks=2, workers=4):
    G = to_networkx(data, to_undirected=True)
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}
    return embeddings


def get_combined_features(data, node2vec_embeddings):
    combined_features = []
    for node_index in range(data.num_nodes):
        node_feature = data.x[node_index]
        node_embedding = torch.tensor(node2vec_embeddings[str(node_index)], dtype=torch.float)
        combined_feature = torch.cat((node_feature, node_embedding), dim=0)
        combined_features.append(combined_feature)
    combined_features = torch.stack(combined_features, dim=0)
    return combined_features

