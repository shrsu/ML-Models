# Importing necessary libraries
import numpy as np                     # For numerical operations
import matplotlib.pyplot as plt       # For plotting graphs
import pandas as pd                   # For handling data in tabular form

# Loading the dataset
dataset = pd.read_csv('Mall_Customers.csv')  # Reading the dataset from a CSV file
X = dataset.iloc[:, [3, 4]].values           # Selecting 'Annual Income' and 'Spending Score' columns

# Importing KMeans from scikit-learn
from sklearn.cluster import KMeans

# Using the Elbow Method to find the optimal number of clusters
wcss = []                                      # List to store Within-Cluster Sum of Squares for each k
for i in range(1, 11):                         # Trying cluster counts from 1 to 10
    kmeans = KMeans(n_clusters=i,             # Initialize KMeans with i clusters
                    init='k-means++',         # Use KMeans++ for smarter centroid initialization
                    random_state=42)          # Fix random state for reproducibility
    kmeans.fit(X)                              # Fit the model to the data
    wcss.append(kmeans.inertia_)               # Append the WCSS (inertia) to the list

# Plotting the Elbow Graph
plt.plot(range(1, 11), wcss)                   # Plot WCSS vs number of clusters
plt.title('The Elbow Method')                  # Graph title
plt.xlabel('Number of clusters')               # X-axis label
plt.ylabel('WCSS')                             # Y-axis label
plt.show()                                     # Display the plot

# Applying KMeans to the dataset with the optimal number of clusters (5 as seen from elbow)
kmeans = KMeans(n_clusters=5,                  # Set number of clusters to 5
                init='k-means++',              # Use KMeans++ initialization
                random_state=42)               # Fix random state
y_kmeans = kmeans.fit_predict(X)              # Fit the model and get cluster labels for each point

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0],               # Select data points in cluster 0
            X[y_kmeans == 0, 1],               # Select corresponding y-axis values
            s=100, c='red', label='Cluster 1') # Plot them with red color

plt.scatter(X[y_kmeans == 1, 0],
            X[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')

plt.scatter(X[y_kmeans == 2, 0],
            X[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')

plt.scatter(X[y_kmeans == 3, 0],
            X[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')

plt.scatter(X[y_kmeans == 4, 0],
            X[y_kmeans == 4, 1],
            s=100, c='magenta', label='Cluster 5')

# Plotting the centroids of all clusters
plt.scatter(kmeans.cluster_centers_[:, 0],     # X coordinates of centroids
            kmeans.cluster_centers_[:, 1],     # Y coordinates of centroids
            s=300, c='yellow', label='Centroids')  # Plot with larger size and yellow color

# Adding chart titles and labels
plt.title('Clusters of customers')             # Title for the plot
plt.xlabel('Annual Income (k$)')               # Label for X-axis
plt.ylabel('Spending Score (1-100)')           # Label for Y-axis
plt.legend()                                   # Show legend
plt.show()                                     # Display the plot
