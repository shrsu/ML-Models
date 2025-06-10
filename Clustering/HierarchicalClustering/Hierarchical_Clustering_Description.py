# Importing essential libraries
import numpy as np                      # For numerical operations and array handling
import matplotlib.pyplot as plt         # For data visualization
import pandas as pd                     # For data manipulation and analysis

# Loading the dataset
dataset = pd.read_csv('Mall_Customers.csv')   # Reads the CSV file into a DataFrame
X = dataset.iloc[:, [3, 4]].values            # Selects only 'Annual Income' and 'Spending Score' columns for clustering

# Creating the dendrogram to help determine the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(                  # Plots the dendrogram
    sch.linkage(X, method='ward')             # Performs hierarchical clustering using Ward's method
)
plt.title('Dendrogram')                       # Adds a title to the plot
plt.xlabel('Customers')                       # Labels the x-axis (each point represents a customer)
plt.ylabel('Euclidean distances')             # Labels the y-axis (shows dissimilarity between clusters)
plt.show()                                    # Displays the dendrogram

# Fitting Agglomerative Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(                # Initializes the Agglomerative Clustering model
    n_clusters=5,                            # Number of clusters to create
    affinity='euclidean',                    # Distance metric used to compute linkage
    linkage='ward'                           # Linkage criterion that minimizes variance within clusters
)
y_hc = hc.fit_predict(X)                     # Fits the model to the data and predicts cluster labels

# Visualizing the resulting clusters
plt.scatter(                                 # Plots all data points belonging to cluster 1
    X[y_hc == 0, 0], X[y_hc == 0, 1],
    s=100, c='red', label='Cluster 1'
)
plt.scatter(                                 # Cluster 2
    X[y_hc == 1, 0], X[y_hc == 1, 1],
    s=100, c='blue', label='Cluster 2'
)
plt.scatter(                                 # Cluster 3
    X[y_hc == 2, 0], X[y_hc == 2, 1],
    s=100, c='green', label='Cluster 3'
)
plt.scatter(                                 # Cluster 4
    X[y_hc == 3, 0], X[y_hc == 3, 1],
    s=100, c='cyan', label='Cluster 4'
)
plt.scatter(                                 # Cluster 5
    X[y_hc == 4, 0], X[y_hc == 4, 1],
    s=100, c='magenta', label='Cluster 5'
)
plt.title('Clusters of customers')           # Title of the scatter plot
plt.xlabel('Annual Income (k$)')             # X-axis label
plt.ylabel('Spending Score (1-100)')         # Y-axis label
plt.legend()                                 # Displays a legend with cluster labels
plt.show()                                   # Renders the final plot
