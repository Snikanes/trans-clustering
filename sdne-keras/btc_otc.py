# coding: utf-8
import matplotlib as mpl
mpl.use('Agg')

import math
import networkx as nx
import pickle as pkl
import pandas as pd
import numpy as np

from itertools import product
from tqdm import tqdm

from matplotlib import pyplot as plt
from core import SDNE

batch_size = 32
dataset = pd.read_csv('../struc2vec/graph/bitcoin-undirected.edgelist', sep=' ', names=['from', 'to'])

g = nx.from_pandas_edgelist(dataset, 'from', 'to')
g = nx.convert_node_labels_to_integers(g, label_attribute='old')

parameter_grid = {'alpha': [2],
                  'l2_param': [1e-3],
                  'pretrain_epochs': [0],
                  'epochs': [5]}

parameter_values = list(product(*parameter_grid.values()))
parameter_keys = list(parameter_grid.keys())

parameter_dicts = [dict(list(zip(parameter_keys, values))) for values in parameter_values]


def one_run(params):
    plt.clf()
    alpha = params['alpha']
    l2_param = params['l2_param']
    pretrain_epochs = params['pretrain_epochs']
    epochs = params['epochs']

    model = SDNE(g, encode_dim=50, encoding_layer_dims=[100, 32],
                 beta=2,
                 alpha=alpha,
                 l2_param=l2_param)
    model.pretrain(epochs=pretrain_epochs, batch_size=32)

    n_batches = math.ceil(g.number_of_edges() / batch_size)
    print("Batch size: {}".format(batch_size))
    print("N-batches: {}".format(n_batches))

    model.fit(epochs=10, log=True, batch_size=batch_size,
              steps_per_epoch=n_batches)

    embedding_name = 'alpha{}-l2_param{}-epochs{}-pre_epochs{}'.format(alpha, l2_param, epochs, pretrain_epochs)
    
    embeddings = model.get_node_embedding()
    return (pd.DataFrame(embeddings), embedding_name)

for params in tqdm(parameter_dicts):
    emb_df, embedding_name = one_run(params)
    print(emb_df.head())
    print(emb_df.info())
    
    node_mapping = nx.get_node_attributes(g, "old")
    emb_df.index = pd.Index(list(map(lambda node: node_mapping[node], emb_df.index.values)))
    emb_df.to_csv("emb/btc/{}.csv".format(embedding_name))