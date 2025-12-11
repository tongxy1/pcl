# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    # 屏蔽开始
    # 更新W
    def _update_W(self, data):
        pdf_vals = [ multivariate_normal.pdf(data,mean=self.Mu[k],cov=np.diag(self.Var[k])) for k in range(self.n_clusters) ]  
        pdf_vals = np.array(pdf_vals).T  # shape: (n_samples, n_clusters)
        weighted_pdfs = pdf_vals * self.pi  # shape: (n_samples, n_clusters)
        sum_weighted_pdfs = np.sum(weighted_pdfs, axis=1, keepdims=True)  # shape: (n_samples, 1)
        self.W = weighted_pdfs / sum_weighted_pdfs  # shape: (n_samples, n_clusters)
        self.W = np.nan_to_num(self.W)  # 处理除0情况
        self.W /= np.sum(self.W, axis=1, keepdims=True)  # 归一化
        self.N = np.sum(self.W, axis=0)  # shape: (n_clusters,)

    # 更新pi
    def _update_pi(self, data):
        pass
        
    # 更新Mu
    def _update_Mu(self, data):
        # data shape: (N, D)
        # W shape: (N, K)
        # Mu shape: (K, D)
        
        #cluster 0
        self.Mu[0]= np.sum(data*self.W[:,0],axis=0)/self.N[0] 
        #cluster 1
        self.Mu[1]= np.sum(data*self.W[:,1],axis=0)/self.N[1] 
        #cluster 2
        self.Mu[2]= np.sum(data*self.W[:,2],axis=0)/self.N[2]
        print(self.Mu)
        
        for k in range(self.n_clusters):
            self.Mu[k]= np.sum(data*self.W[:,k],axis=0)/self.N[k] #(n_samples,n_features) * (n_samples,1)
        print(self.Mu)

        self.Mu = (self.W.T @ data) / self.N[:, np.newaxis]
        print(self.Mu)


    # 更新Var
    def _update_Var(self, data):
        pass    

    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        random.seed(42)
        n_samples, n_features = data.shape

        # step 0：初始化参数
        self.Mu = np.array([data[random.randint(0, n_samples - 1)] for _ in range(self.n_clusters)])
        self.Var = np.array([np.ones(n_features) for _ in range(self.n_clusters)])
        self.W = np.zeros((n_samples, self.n_clusters))
        self.pi = np.array([1.0 / self.n_clusters for _ in range(self.n_clusters)])            

        for iter in range(self.max_iter):
            # E-step 
            self._update_W(data)
            # M-step
            self._update_Mu(data)
            self._update_Var(data)
            self._update_pi(data)           
            

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        pass
        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.savefig('GMM_data.png', dpi=300)
    # plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

