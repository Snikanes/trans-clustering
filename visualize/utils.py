import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import networkx as nx
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def enhance_with_clusterings(embeddings, eps=0.1, min_samples=10, all_cols=False):
    cols = embeddings if all_cols else embeddings.filter(items=['X', 'Y'])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cols.as_matrix())
    series = pd.Series(clustering.labels_, dtype='int32')
    cluster_df = pd.DataFrame({'Id': embeddings.index.values, 'Cluster': series})
    cluster_df.set_index('Id', inplace=True)
    embeddings = embeddings.join(cluster_df)
    return embeddings
    
def get_norm(num_colors=0, logarithmic=False, vmin=-1.0):
    norm_type = colors.LogNorm if logarithmic else colors.Normalize
    return norm_type(vmin=vmin, vmax=num_colors - 1)

def show_clusters(embeddings, cmap='gnuplot', cluster_labels=False, save=False, save_prefix='img'):
    clusters = embeddings['Cluster'].unique()
    norm = get_norm(num_colors=len(clusters))
    axes = embeddings.plot.scatter(x='X', y='Y', c='Cluster', colormap=cmap, norm=norm, figsize = (30,15))
    if cluster_labels:
        for i, cluster in enumerate(clusters):
            axes.annotate(cluster, embeddings.loc[embeddings['Cluster']==cluster, ['X','Y']].mean(), horizontalalignment='center', verticalalignment='center', size=20)
    fig = axes.get_figure()
    if save:
        fig.savefig('{}_cluster.png'.format(save_prefix), bbox_inches='tight')
    fig.show()
    
def show_implicit_clusters(embeddings, cmap='gnuplot', cluster_labels=False, implicit_clusters=None, save=False, save_prefix='img'):
    
    def find_in_listof_list(l, element):
        for i in range(len(l)):
            if element in l[i]:
                return i
        return -1
    
    norm = get_norm(num_colors=len(implicit_clusters) + 1)
    emb_copy = embeddings.copy(deep=True)
    emb_copy['Implicit Cluster'] = pd.Series(list(map(lambda node: find_in_listof_list(implicit_clusters, node), emb_copy.index.values)))
    print(emb_copy.head())
    
    axes = emb_copy.plot.scatter(x='X', y='Y', c='Implicit Cluster', colormap=cmap, norm=norm, figsize = (30,15))        
    fig = axes.get_figure()
    if save:
        fig.savefig('{}_cluster.png'.format(save_prefix), bbox_inches='tight')
    fig.show()
    
def get_cluster_members(df, cluster):
    return df[df['Cluster'] == cluster]['ids'].values

def get_cluster_members_other(df, cluster):
    return df[df['Cluster'] == cluster].index.values

def draw_subgraph(G, source_node, with_labels=True, ax=None, pos=None):
    if ax is None:
        fig, ax = plt.figure(40, 15)
        pos = nx.spring_layout(G)
    val_map = {source_node: 0.8}   
    values = [val_map.get(node, 0.25) for node in G.nodes()]
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=with_labels, cmap=plt.get_cmap('coolwarm'), node_color=values, node_size=15, width=0.7)
    
def create_egograph_comparison(G, nodes, index, save=False, savepath=None, radius=1):
    fig, ax = plt.subplots(1, len(nodes), figsize=(40, 15))
    
    for i, node in enumerate(nodes):
        ego_graph = nx.ego_graph(G, node, radius=radius)
        pos = nx.spring_layout(ego_graph)
        
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)
        draw_subgraph(ego_graph, node, with_labels=False, ax=ax[i], pos=pos)
    
    if save:
        fig.savefig('{}_{}.png'.format(savepath, index), bbox_inches='tight')
    
def get_closure_links(G, ego_node):
    return list(filter(lambda edge: not (edge[0] == ego_node or edge[1] == ego_node), G.edges()))

def gamma(G, ego_node):
    closure_links = get_closure_links(G, ego_node)
    involved_nodes = set()
    for link in closure_links:
        involved_nodes.add(link[0])
        involved_nodes.add(link[1])
    
    M = 2 * len(closure_links)  
    t = len(closure_links)
    m = (1 + np.sqrt(1 + 8*t)) / 2
    k_t = len(involved_nodes)
    if t == 0 and k_t == 0:
        return 0
    elif t == 1:
        return 0.5
    return (M - k_t) / (M - m)

def get_degree(G, node):
    return len(G.edges(node))

def extract_graph_info(graph):
    ids = []
    ccs = []
    gammas = []
    degrees = []
    
    for node in tqdm(graph.nodes):
        ego_graph = nx.ego_graph(graph, node)
        gamma_val = gamma(ego_graph, node)
        cc = nx.clustering(ego_graph, node)
        degree = get_degree(graph, node)

        gammas.append(gamma_val)
        ids.append(node)
        ccs.append(cc)
        degrees.append(degree)
    
    return ids, ccs, gammas, degrees

def show_embedding_plot(embeddings, color_col=None, cmap='gnuplot', norm=None, save=False, savefile=None):
    axes = embeddings.plot.scatter(x='X', y='Y', c=color_col, colormap=cmap, norm=norm, figsize = (30,15))
    fig = axes.get_figure()
    if save:
        fig.savefig(savefile, bbox_inches='tight')
    fig.show()

# Takes a dataframe of format x, y, gammas, ccs, degrees and outputs plots for each of (gammas, ccs, degrees) 
def show_embedding_plots(embeddings, cmap='gnuplot', save=False, save_prefix='img'):
    log_norm = get_norm(num_colors=embeddings['Node Degree'].max(), logarithmic=True, vmin=1.0)
    
    show_embedding_plot(embeddings, color_col='Cluster Coefficient', cmap=cmap, save=save, savefile='{}_cc.png'.format(save_prefix))
    show_embedding_plot(embeddings, color_col='Gamma', cmap=cmap, save=save, savefile='{}_gamma.png'.format(save_prefix))
    show_embedding_plot(embeddings, color_col='Node Degree', cmap=cmap, norm=log_norm, save=save, savefile='{}_degree.png'.format(save_prefix))