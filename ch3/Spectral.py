import numpy as np
from scipy.spatial import KDTree

class Spectral:
  def __init__(self):
    pass
  def predict(self, data):
    n_samples, n_features = data.shape
    
    #创建W矩阵
    W = np.zeros( (n_samples, n_samples))
    
    kdtree = KDTree(data,leaf_size=2)    
    dist, idx = kdtree.query(data, k=10)
    for i, j, d in enumerate(zip(idx, dist)):
      W[i,j] = 1/(d+1e-10)


      
          
    
