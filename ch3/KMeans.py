# 文件功能： 实现 K-Means 算法

import numpy as np
import random

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        
        # random.seed(42)
        n_samples, n_features = data.shape

        # step 0：初始化参数        
        # (K, D)
        self.C = np.array([data[random.randint(0, n_samples - 1)] for _ in range(self.k_)]) 

        for iter in range(self.max_iter_):
            # E-step 
            #(N, K)
            dist = [ np.linalg.norm( data-self.C[k], axis=1 ) for k in range(self.k_) ]  
            dist = np.array(dist).T     
            #(N,)
            cat = np.argmin(dist, axis=1)    
            
            # M-step
            # first implement
            #
            # idx = [[]]*2
            # idx[0] = np.where(cat==0)
            # idx[1] = np.where(cat==1)             
            # C_new = np.empty((2, n_features))
            # C_new[0] = np.mean(data[idx[0]], axis=0) 
            # C_new[1] = np.mean(data[idx[1]], axis=0) 
            # print(C_new)

            # second implement
            #
            idx = [ np.where( cat == i) for i in range(self.k_) ]             
            C_new = np.array([ np.mean(data[idx[i]], axis=0)   for i in range(self.k_) ])
            # print(C_new)
            

            if np.all( np.abs(C_new-self.C) < self.tolerance_):
                break            
            
            self.C = C_new            
    
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        
        #(N, K)
        dist = [ np.linalg.norm( p_datas-self.C[k], axis=1 ) for k in range(self.k_) ]  
        dist = np.array(dist).T     
        #(N,)
        result = np.argmin(dist, axis=1)    
        
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    cat = k_means.predict(x)
    print("Ck:\n", k_means.C)
    print(cat)
 
    # 可视化聚类结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))    
    cluster1 = np.where(cat==0)
    cluster2 = np.where(cat==1)
    plt.scatter(x[cluster1][:,0], x[cluster1][:,1], c="red", marker = "o",s=50)
    plt.scatter(x[cluster2][:,0], x[cluster2][:,1], c="green", marker = "x", s=50)
    plt.savefig('k_means_result.png', dpi=300)
    #
    #sklearn implement for comparing result
    #
    from sklearn.cluster import KMeans    
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_
    print("sklearn Ck:\n", centers)
    print("sklearn labels:\n", labels)
    