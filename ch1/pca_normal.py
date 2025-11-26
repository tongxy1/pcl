# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    center = np.mean(data, axis=0)
    data_zero_mean = data - center

    Cov = data_zero_mean.T @ data_zero_mean
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)
    
    # eigenvectors1, S, Vh = np.linalg.svd(data.T)
    # eigenvalues1 = np.square(S)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 加载原始点云
    with open('/RobotGrasp/pcl/ch1/modelnet40_normal_resampled/'
              'modelnet40_shape_names.txt') as f:
        a = f.readlines()
    for i in a:
        point_cloud_pynt = PyntCloud.from_file(
            '/RobotGrasp/pcl/ch1/modelnet40_normal_resampled/'
            '{}/{}_0001.txt'.format(i.strip(), i.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.xyz
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    
    # TODO: 同时显示点云，PCA主方向
    R = v
    t = [0, 0, 0]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame.rotate(R, center=[0, 0, 0])
    frame.translate(t)
    frame.scale(5.0, center=[0, 0, 0])   # 整体放大
    o3d.visualization.draw_geometries([point_cloud_o3d, frame]) # 同时显示点云和PCA主方向
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2    
    # 屏蔽开始
    
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数    
    points = np.asarray(point_cloud_o3d.points)    
    for i in range(points.shape[0]):        
        # 对点 i 找 30 个邻居    
        _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], 30)
        # 邻域点
        neighbor_pts = points[idx, :]        
        # PCA特征值分解
        eigenvalues, eigenvectors = PCA(neighbor_pts)        
        normals.append(eigenvectors[:, -1])# 法向量 = 最小特征值对应的特征向量

    # 屏蔽结束

    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    
    # 显示带法向量的点云
    # 统一法线方向（例如朝向原点）
    point_cloud_o3d.orient_normals_towards_camera_location(camera_location=(0, 0, 0))
    # 可视化
    o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normal=True)

    # o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
