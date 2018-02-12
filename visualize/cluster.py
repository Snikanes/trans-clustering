from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import pandas as pd

in_file = '/Users/eirikvikanes/Dropbox/Skolearbeid/Master/struc2vec/emb/bitcoin.emb'

# Needed for KMeans
num_clusters = 6

# Plots the cluster in two-dimensional space
def plot_cluster(cluster, data):
    #labels = cluster.labels_    
    print(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.show()

# Cluster using K-Means
def kmeans(data, num_clusters):
    return KMeans(n_clusters=num_clusters).fit(data)

# Cluster using DBScan
def dbscan(data):
    return DBSCAN(eps=0.01, min_samples=17).fit(data)

# Cluster using HAC
def agglomerative(data, num_clusters):
    return AgglomerativeClustering(n_clusters=num_clusters).fit(data)
    
def tsne(data):
    return TSNE().fit(data) 
# Retrieve embeddings
embeddings = pd.read_csv(in_file, sep=' ', header=0)

# Scale the features
scaler = preprocessing.MinMaxScaler().fit(embeddings)
scaled_embeddings = scaler.transform(embeddings)

# Create kmeans and fit to data
#kmeans_fit = kmeans(scaled_embeddings, num_clusters)
#plot_cluster(kmeans_fit, scaled_embeddings)
#
## Create dbscan and fit to data
#dbscan_fit = dbscan(scaled_embeddings)
#plot_cluster(dbscan_fit, scaled_embeddings)
#
## Create HAC and fit to data
#agg_fit = agglomerative(scaled_embeddings, num_clusters)
#plot_cluster(agg_fit, scaled_embeddings)

tsne_fit = tsne(scaled_embeddings)
plot_cluster(tsne_fit, scaled_embeddings)