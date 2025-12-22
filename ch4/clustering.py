# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    best_i, best_j, best_k = None, None, None
    best_d = None
    best_n = None    
    n_inliers_max = 0    
    iterate_num = 500
    d_threshold = 0.3
    for _ in range(iterate_num):
        i,j,k = np.random.choice(N, 3, replace=False)
        n = np.cross(data[j]-data[i], data[k]-data[i])
        n = n / np.linalg.norm(n)
        d = (data-data[i]) @ n.reshape(3,1)
        idx = np.where(np.abs(d) < d_threshold)
        n_inliers = idx[0].shape[0]
        if n_inliers > n_inliers_max:
            n_inliers_max = n_inliers
            best_i, best_j, best_k = i, j, k
            best_n = n
            best_d = d
            best_idx = idx
    
    segmengted_cloud = data[np.where(np.abs(d) >= d_threshold)[0], :]  
    #在 RANSAC 找到内点集合之后，用所有内点做一次最小二乘（LSQ）平面拟合
    # inlier_points = data[best_idx]
    # centroid = np.mean(inlier_points, axis=0)
    # cov = np.cov(inlier_points - centroid, rowvar=False)
    # eigvals, eigvecs = np.linalg.eig(cov)
    # min_eigval_index = np.argmin(eigvals)
    # best_n = eigvecs[:, min_eigval_index]
    # # 根据拟合的平面模型，滤除地面点
    # d = (data - centroid) @ best_n
    # segmengted_cloud = data[np.abs(d) >= d_threshold]    

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始


    # 屏蔽结束

    return clusters_index

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
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(origin_points)
        o3d.visualization.draw_geometries([pcd])
        
        # 滤除地面点
        segmented_points = ground_segmentation(data=origin_points)

        pcd.points = o3d.utility.Vector3dVector(segmented_points)
        o3d.visualization.draw_geometries([pcd])

        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
