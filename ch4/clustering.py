# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os, time
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
from scipy.spatial import KDTree


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    N, _ = data.shape    
  
    # 初始化参数
    n_inliers_max = 0    
    iterate_num_max = 500
    d_threshold = 0.20
  
    for curr_num in range(iterate_num_max):
        # 随机采样3个点，确定一个平面
        i,j,k = np.random.choice(N, 3, replace=False)
        
        # 计算平面法向量和点到平面的距离
        n = np.cross(data[j]-data[i], data[k]-data[i])
        n = n / np.linalg.norm(n)
        d = (data-data[i]) @ n.reshape(3,1)

        # 统计内点数量
        idx = np.where(np.abs(d) < d_threshold)
        n_inliers = idx[0].shape[0]

        # 通过比较和上一次的内点数量，更新最优平面模型
        if n_inliers > n_inliers_max:
            n_inliers_max = n_inliers            
            best_idx = idx[0]            
            
            # 自适应调整迭代次数
            w = n_inliers_max / N
            iterative_prob = 0.99
            iterate_num = int(np.log(1 - iterative_prob) / np.log(1 - w**3) + 1)
            if iterate_num < curr_num:
                print(f'RANSAC early stop at iter:{curr_num+1}/prob_num:{iterate_num}/max_num:{iterate_num_max}')
                break         
    
    #在 RANSAC 找到内点集合之后，用所有内点做一次最小二乘（LSQ）平面拟合
    inlier_points = data[best_idx, :]
    P = np.ones((inlier_points.shape[0], 4))
    P[:, :3] = inlier_points    
    eigvals, eigvecs = np.linalg.eigh( P.T @ P)    
    min_eigvec = eigvecs[:, np.argmin(eigvals)]
    n = min_eigvec[:3]
    n = n / np.linalg.norm(n)    
   
    # 根据最终拟合的平面模型，滤除地面点
    ref_pt = np.array([0,0,-min_eigvec[3]/(min_eigvec[2]+1e-10)])  # 平面上距离原点最近的点    
    d = (data-ref_pt.reshape(1,3)) @ n.reshape(3,1)    
    ground_idx = np.where(np.abs(d) < d_threshold)[0]
    non_ground_idx = np.where(np.abs(d) >= d_threshold)[0]    
    segmengted_cloud = data[non_ground_idx, :]    
    

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, non_ground_idx, ground_idx 

    
def my_dbscan(data, r=0.5, minPts=10):    
    N = data.shape[0]

    kdtree = KDTree(data,leafsize=3)    
    idx = kdtree.query_ball_point(data, r=r)    

    n_clusters = 0
    labels = -1 * np.ones(N, dtype=int) 
    
    visited = np.zeros(N, dtype=bool)   
    for i in range(N):
        
        if visited[i]:
            continue
        visited[i] = True

        count = len(idx[i])
        if count < minPts: # 该点为Noise Point
            continue
        else: # count >= minPts # 该点为Core Point，创建新cluster            
            # labels[i] = n_clusters # 分配新cluster标签

            cluster_set = set(idx[i])  # 该点邻域点全部加入该cluster索引集合中，准备逐一打标签
            # cluster_set.discard(i) # 该cluster索引集合中去除自身点，已经打过标签

            while cluster_set:
                j = cluster_set.pop() # 从该cluster索引集合中任意取一个点

                labels[j] = n_clusters # 为该点打上当前聚类标签

                if not visited[j]:
                    visited[j] = True

                    count = len(idx[j])
                    if count >= minPts:
                        cluster_set.update(idx[j]) # 该点为Core Point，其邻域点也加入该cluster索引集合中
                        # cluster_set.discard(j) # 移除自身点j，这是idx[j]中包含的j。虽然j点已经pop移除，这里还需要要二次移除。

            n_clusters += 1 # 下一个聚类标签    

    return labels

def my_dbscan2(data, r=0.5, minPts=10):    
    N = data.shape[0]

    kdtree = KDTree(data,leafsize=3)    
    idx = kdtree.query_ball_point(data, r=r)    

    n_clusters = 0
    labels = -1 * np.ones(N, dtype=int) # -1表示不属于任何cluster，循环结束后若仍为-1，即表示噪声点
    isLabeled = np.zeros(N, dtype=bool) 

    # 大循环，找到新cluster入口
    for i in range(N):        
        if not isLabeled[i]:  # 只处理未打标签的点

            count = len(idx[i])            
            if count >= minPts: # 该点为Core Point，即为新cluster入口
                labels[i] = n_clusters 
                isLabeled[i] = True
                cluster_set = set(idx[i])  # 该点邻域点全部加入该cluster索引集合中，准备逐一打标签                
                
                # 小循环开始，扩展当前cluster
                while cluster_set: 
                    j = cluster_set.pop() # 从该cluster索引集合中任意取一个点                    
                    
                    if not isLabeled[j]:
                        labels[j] = n_clusters # 为该点打上当前聚类标签
                        isLabeled[j] = True
                        
                        count = len(idx[j])
                        if count >= minPts: # 该点为Core Point，其邻域点加入当前cluster，实现聚类扩展
                            cluster_set.update(idx[j]) 
                
                # 小循环结束，回到大循环继续寻找下一个cluster
                n_clusters += 1 

    return labels

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始    
    now = time.time()    
    my_clusters_index = my_dbscan2(data, r=0.5, minPts=10)
    print(f'my DBSCAN time: {time.time() - now:.3f} s')

   
    now = time.time()
    dbscan = cluster.DBSCAN(eps=0.5, min_samples=10)
    dbscan.fit(data)
    ski_clusters_index = dbscan.labels_
    print(f'sklearn DBSCAN time: {time.time() - now:.3f} s')

    # 比较自己实现的DBSCAN和sklearn的DBSCAN结果是否一致
    my_count_dict = {}
    for i in range(np.max(my_clusters_index)+1):
        my_count = np.sum(my_clusters_index == i)                
        my_count_dict.update({i:my_count})        
    
    ski_count_dict = {}
    for i in range(np.max(ski_clusters_index)+1):
        ski_count = np.sum(ski_clusters_index == i)                        
        ski_count_dict.update({i:ski_count})

    print("-----------------------------------")
    print('Number of clusters:')
    print(f'My DBSCAN found {len(my_count_dict)} clusters.')
    print(f'Sklearn DBSCAN found {len(ski_count_dict)} clusters.')
    print("-----------------------------------")
    print('Cluster size comparison:')    
    print('Cluster ID : My DBSCAN Size | Sklearn DBSCAN Size')
    differences = 0
    for cluster_id in range(max(len(my_count_dict), len(ski_count_dict))):
        my_size = my_count_dict.get(cluster_id, 0)
        ski_size = ski_count_dict.get(cluster_id, 0)
        #如果大小不一样，则用不同颜色打印
        if my_size != ski_size:
            print(f'\033[91m     {cluster_id}     :      {my_size}      |        {ski_size}   \033[0m')
            differences += 1
        else:
            print(f'     {cluster_id}     :      {my_size}      |        {ski_size}')   
            
    print("--------------------------------------------")
    print(f"Total different clusters num: {differences} / {max(len(my_count_dict), len(ski_count_dict))}")
    print("-------------------------------------------")    
    
        # 屏蔽结束

    return my_clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def plot_clusters_open3d(data, cluster_index):    
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    rgb_colors = np.array([list(int(colors[i][j+1:j+3], 16) for j in (0, 2, 4)) for i in range(colors.shape[0])]) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors[cluster_index])
    o3d.visualization.draw_geometries([pcd])



def main():
    root_dir = 'ch4/data/' # 数据集路径
    cat = os.listdir(root_dir)
    # cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        # 可视化原始点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(origin_points)
        # o3d.visualization.draw_geometries([pcd])
        
        # 滤除地面点
        segmented_points, non_ground_idx, ground_idx = ground_segmentation(data=origin_points)
        
        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(origin_points[ground_idx, :])
        pcd_ground.colors = o3d.utility.Vector3dVector(np.ones((ground_idx.shape[0], 3)) * np.array([0.0, 0.0, 1.0]))  # 蓝色表示地面点
        
        pcd_other = o3d.geometry.PointCloud()
        pcd_other.points = o3d.utility.Vector3dVector(origin_points[non_ground_idx, :])        
        o3d.visualization.draw_geometries([pcd_ground, pcd_other])      
        
        cluster_index = clustering(segmented_points)

        # plot_clusters(segmented_points, cluster_index)
        plot_clusters_open3d(segmented_points, cluster_index)


if __name__ == '__main__':
    main()
