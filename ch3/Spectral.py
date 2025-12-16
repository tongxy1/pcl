import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans    
import matplotlib.pyplot as plt

class Spectral:
  def __init__(self, n_clusters):
     self.n_clusters = n_clusters

  def fit(self, data):
     pass

  def predict(self, data):
    n_samples, n_features = data.shape
    
    # 创建 Weighted adjacency matrix / similarity matrix
    W = np.zeros( (n_samples, n_samples))
    
    kdtree = KDTree(data,leafsize=2)    
    dist, idx = kdtree.query(data, k=10)
    for i, (j, d) in enumerate(zip(idx, dist)):
      W[i, j[1:]] = 1/(d[1:]+1e-10)
      W[j[1:], i] = 1/(d[1:]+1e-10)
    
    # 创建Normalized Laplacian Matrix
    d = np.sum(W, axis=1)    
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    Lsym = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt

    # 创建 k columns smallest eigenvector matrix
    eigenvalues, eigenvectors = np.linalg.eigh(Lsym)     
    sort = np.argsort(eigenvalues)
    Vk = eigenvectors[:,sort[:self.n_clusters]]

    # 对Vk进行Kmeans分类
    kmeans = KMeans(n_clusters= self.n_clusters )
    labels = kmeans.fit_predict(Vk)

    return labels

if __name__ == "__main__":
    from sklearn import datasets

    n_samples = 1500
    data, ground_labels = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
    
    spectral = Spectral(n_clusters=2)
    output_labels = spectral.predict(data)    
    
    for labels in (ground_labels, output_labels):
      n_cluster1 = len(np.argwhere(labels == 0))
      n_cluster2 = len(np.argwhere(labels == 1))      
      print(f"n_cluster1: {n_cluster1}")
      print(f"n_cluster2: {n_cluster2}")
    
    #对齐标签
    if ground_labels[0] != output_labels[0]:
       output_labels = np.where(output_labels == 0, 1, 0)
       
    false_labels= np.argwhere(ground_labels != output_labels)
    print(f"ratio: {len(false_labels)}/{n_samples}")

    plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    plt.scatter(data[:, 0], data[:, 1], c=output_labels, s=5, cmap='viridis')    
    plt.savefig('spectral_result.png', dpi=300)

    
      
          
    
