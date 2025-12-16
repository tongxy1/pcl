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
        # 计算每个数据点在每个（多维）高斯分布下的概率密度值
        # ( N, K )
        pdf_vals = [ multivariate_normal.pdf( data, mean=self.Mu[k], cov=np.diag(self.Var[k]) ) for k in range(self.n_clusters) ]  
        pdf_vals = np.array(pdf_vals).T          
        
        # 计算每个数据点属于每个簇的概率
        # ( N, K ) = ( N, K ) * ( 1, K )
        weighted_pdfs = pdf_vals * self.pi  
        
        # 计算每个数据点在所有簇下的加权概率密度值之和
        #  (N, 1)
        sum_weighted_pdfs = np.sum(weighted_pdfs, axis=1, keepdims=True)  
        
        # 计算每个数据点属于每个簇的概率
        # ( N, K ) = ( N, K ) / ( N, 1 )
        self.W = weighted_pdfs / sum_weighted_pdfs  
        self.W = np.nan_to_num(self.W)  # 处理除0情况
        self.W /= np.sum(self.W, axis=1, keepdims=True)  # 归一化
        
        # 计算每个簇的有效样本数
        #  (K,)
        self.N = np.sum(self.W, axis=0) 
  
        
    # 更新Mu
    def _update_Mu(self, data):
        # data shape: (N, D)
        # W shape: (N, K)
        # Mu shape: (K, D)
        
        # #第1种写法        
        # #
        # #cluster 0
        # self.Mu[0]= np.sum( data * self.W[:,0][:,None], axis=0) / self.N[0] 
        # #cluster 1
        # self.Mu[1]= np.sum( data * self.W[:,1][:,None], axis=0) / self.N[1] 
        # #cluster 2
        # self.Mu[2]= np.sum( data * self.W[:,2][:,None], axis=0) / self.N[2]
        # print(self.Mu)
        
        # #第2种写法
        # #
        # for k in range(self.n_clusters):
        #     self.Mu[k]= np.sum(data * self.W[:,k][:,None], axis=0) / self.N[k] #(n_samples,n_features) * (n_samples,1)
        # print(self.Mu)

        #第3种写法
        # #
        self.Mu = (self.W.T @ data) / self.N[:, np.newaxis]
        # print(self.Mu)


    # 更新Var
    def _update_Var(self, data):
        
        # #第1种写法
        # #
        # #cluster 0
        # self.Var[0]= np.sum( np.square(data-self.Mu[0]) * self.W[:,0][:,None], axis=0 ) / self.N[0] 
        # #cluster 1
        # self.Var[1]= np.sum( np.square(data-self.Mu[1]) * self.W[:,1][:,None], axis=0 ) / self.N[1] 
        # #cluster 2
        # self.Var[2]= np.sum( np.square(data-self.Mu[2]) * self.W[:,2][:,None], axis=0 ) / self.N[2] 
        # print(self.Var)

        #第2种写法
        #   
        for k in range(self.n_clusters):
            self.Var[k]= np.sum( np.square(data - self.Mu[k]) * self.W[:,k][:,None], axis=0 ) / self.N[k] 
        # print(self.Var)

        #第3种写法
        # # (N, K, D) 
        # var_data = data[:, np.newaxis, :] - self.Mu[np.newaxis, :, :]
        # np.square(var_data, out=var_data)  # 就地平方，节省内存
        # # (N, K, D) * (N, K, 1) -> (N, K, D)
        # weighted_var_data = var_data * self.W[:, :, np.newaxis]  
        # # (K, D)
        # self.Var = np.sum(weighted_var_data, axis=0) / self.N[:, np.newaxis]  
        # print(self.Var)
        
    # 更新pi
    def _update_pi(self, data):
        self.pi = self.N/data.shape[0]  # (K,)
        
    # 屏蔽结束

    
    def fit(self, data):
        # 作业3
        # 屏蔽开始

        random.seed(42)
        n_samples, n_features = data.shape

        # step 0：初始化参数        
        # (K, D)
        self.Mu = np.array([data[random.randint(0, n_samples - 1)] for _ in range(self.n_clusters)])  
        # (K, D)       
        self.Var = np.array([np.ones(n_features) for _ in range(self.n_clusters)])  
        # (N, K)
        self.W = np.zeros((n_samples, self.n_clusters))  
        # (K,)           
        self.pi = np.array([1.0 / self.n_clusters for _ in range(self.n_clusters)]) 

        self.History_Mu = []
        for iter in range(self.max_iter):
            # 记录以便看可视化收敛情况
            self.History_Mu.append(self.Mu)                    
            # E-step 
            self._update_W(data)
            # M-step
            self._update_Mu(data)
            self._update_Var(data)
            self._update_pi(data)        

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        
        self._update_W(data)
        return np.argmax(self.W, axis=1)
        
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
    # plt.savefig('GMM_data.png', dpi=300)
    # plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    print("Mu:\n", gmm.Mu)
    print("Var:\n", gmm.Var)
    print("pi:\n", gmm.pi)

    cat = gmm.predict(X)    
    print(cat)
    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])    
    plt.scatter(X[:, 0], X[:, 1], c=cat, s=5, cmap='viridis')
    hisMu = np.array(gmm.History_Mu)

    plt.plot(hisMu[:,0,0], hisMu[:,1,1],color='red')
    plt.plot(hisMu[:,1,0], hisMu[:,1,1],color='green')
    plt.plot(hisMu[:,2,0], hisMu[:,2,1],color='blue')
    plt.savefig('GMM_result.png', dpi=300)
    # plt.show()  

    # # 可视化高斯分布轮廓
    # ax = plt.gca()
    # for k in range(gmm.n_clusters):
    #     mu = gmm.Mu[k]
    #     var = gmm.Var[k]
    #     eigenvalues, eigenvectors = np.linalg.eig(np.diag(var))
    #     order = eigenvalues.argsort()[::-1]
    #     eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    #     angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    #     for nsig in range(1, 4):
    #         ell = Ellipse(xy=mu,
    #                       width=2 * nsig * np.sqrt(eigenvalues[0]),
    #                       height=2 * nsig * np.sqrt(eigenvalues[1]),
    #                       angle=angle,
    #                       edgecolor='red',
    #                       fc='None',
    #                       lw=1.5,
    #                       ls='--')
    #         ax.add_patch(ell)
    # plt.savefig('GMM_contours.png', dpi=300)
    # # plt.show()

    #采用sklearn的GMM进行对比
    from sklearn.mixture import GaussianMixture
    gmm_sklearn = GaussianMixture(n_components=3, covariance_type='diag', max_iter=50, random_state=42)
    gmm_sklearn.fit(X)
    print("Sklearn Mu:\n", gmm_sklearn.means_)
    print("Sklearn Var:\n", gmm_sklearn.covariances_)
    print("Sklearn pi:\n", gmm_sklearn.weights_)
    cat_sklearn = gmm_sklearn.predict(X)
    print(cat_sklearn) 

    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])     
    plt.scatter(X[:, 0], X[:, 1], c=cat_sklearn, s=5, cmap='viridis')
    plt.savefig('GMM_sklearn_result.png', dpi=300)
    # plt.show()    
    
    # 可视化高斯分布轮廓
    # ax = plt.gca()
    # for k in range(gmm_sklearn.n_components):       
    #     mu = gmm_sklearn.means_[k]
    #     var = gmm_sklearn.covariances_[k]
    #     eigenvalues, eigenvectors = np.linalg.eig(np.diag(var))
    #     order = eigenvalues.argsort()[::-1]
    #     eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    #     angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    #     for nsig in range(1, 4):
    #         ell = Ellipse(xy=mu,
    #                       width=2 * nsig * np.sqrt(eigenvalues[0]),
    #                       height=2 * nsig * np.sqrt(eigenvalues[1]),
    #                       angle=angle,
    #                       edgecolor='red',
    #                       fc='None',
    #                       lw=1.5,
    #                       ls='--')
    #         ax.add_patch(ell)
    # plt.savefig('GMM_sklearn_contours.png', dpi=300)
    # # plt.show()


    

