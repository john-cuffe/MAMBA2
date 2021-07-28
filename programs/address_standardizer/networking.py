import networkx as nx
import pandas as pd
from sklearn.cluster import AffinityPropagation

"""
transforms a dataframe of matches into a graph, with match scores as weights
df_matches: dataframe of matches [address1, address2, score] 
"""
def match_network(df_matches):
    G = nx.Graph()
    G.add_weighted_edges_from(df_matches.values.tolist())
    # if input isn't df, but something like a list of lists, the code is 
    # basically the same
    return G

## if networkx.to_pandas_adjacency was *working* this wouldn't be necessary
# important thing here is that ORDER IS PRESERVED - can index back into nodes
"""
makes a pandas dataframe that represents the adjacency matrix of an input graph
-- columns & indices are nodes, order is preserved after AffinityPropagation
"""
def make_adjacency_matrix(network):
    adj_matrix = pd.DataFrame(0, columns = network.nodes(), index = network.nodes())
    for (a, b) in network.edges():
        weight = network[a][b]['weight']
        adj_matrix.at[a, b] = weight
        adj_matrix.at[b, a] = weight
    return adj_matrix

"""
uses sklearn's AffinityPropagation implementation to find exemplars in a network
of addresses where edges are matches (can be weighted)
network: the overall network
subgraphs: if True, will run AffinityPropagation on each unique subgraph separately
           -- False will generally lead to a single address per cluster
output: a mapping from each node to its cluster center/exemplar
"""
def disentangle(network, subgraphs = True):
    if subgraphs:
        graphs = [network.subgraph(c).copy() for c in \
            nx.connected_components(network) if len(c) > 1]
    else:
        graphs = [network] if len(network.nodes()) > 1 else []
    final_map = {}
    for g in graphs:
        adj_matrix = make_adjacency_matrix(g)
        # will break if there is only one node passed in
        model = AffinityPropagation(affinity='precomputed').fit(adj_matrix)

        ids = g.nodes()
        # the database ids of the deduplicated records; the "centers" of the clusters
        centers = [ids[index] for index in model.cluster_centers_indices_]
        # the mapping of each record id to its centralized record's id
        mapping = {ids[index]: centers[cluster] for index, cluster in enumerate(model.labels_)}
        final_map.update(mapping)
    # centers is just final_map.values()
    return final_map