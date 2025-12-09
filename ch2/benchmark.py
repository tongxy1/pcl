# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

import octree as octree
import kdtree as kdtree

from result_set import KNNResultSet, RadiusNNResultSet

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

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001

    k = 8
    radius = 1

    # root_dir = '/Users/renqian/cloud_lesson/kitti' # 数据集路径
    root_dir = '/RobotGrasp/pcl/ch2/kitti'
    cat = os.listdir(root_dir)
    iteration_num = len(cat)


    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        query = db_np[0,:]

        print("octree --------------")        
        construction_time_sum = 0
        knn_time_sum = 0
        radius_time_sum = 0        

# octree创建
        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        depth = [0]
        max_depth = [0]
        octree.traverse_octree(root, depth, max_depth)
        print("octree max depth: %d" % max_depth[0])
        

# octree knn search
        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

# octree radius search
        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t


        print("Octree: build %.3f, knn %.3f, radius %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num))
        
        print("kdtree --------------")
        construction_time_sum = 0
        knn_time_sum = 0
        radius_time_sum = 0        
# 自实现kdtree 创建
        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t
        
        depth = [0]
        max_depth = [0]
        kdtree.traverse_kdtree(root, depth, max_depth)
        print("kdtree max depth: %d" % max_depth[0])

# 自实现kdtree knn search
        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

# 自实现kdtree radius search
        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        print("Kdtree: build %.3f, knn %.3f, radius %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num))

        print("sci kdtree --------------")
        sci_construction_time_sum = 0
        sci_knn_time_sum = 0
        sci_radius_time_sum = 0
            
        from scipy.spatial import KDTree

# scipy kdtree 创建
        begin_t = time.time()
        tree = KDTree(db_np, leafsize=leaf_size)
        sci_construction_time_sum += time.time() - begin_t

# scipy kdtree knn search
        begin_t = time.time()
        dist, idx = tree.query(query, k=k)
        sci_knn_time_sum += time.time() - begin_t

# scipy kdtree radius search
        begin_t = time.time()
        neighbors = tree.query_ball_point(query, r=radius)
        sci_radius_time_sum += time.time() - begin_t
        print("Scipy KDTree: build %.3f, knn %.3f, radius %.3f" % (sci_construction_time_sum*1000/iteration_num,
                                                                 sci_knn_time_sum*1000/iteration_num,
                                                                 sci_radius_time_sum*1000/iteration_num))

        print("open3d kdtree --------------")
        construction_time_sum = 0
        knn_time_sum = 0
        radius_time_sum = 0
            
        import open3d as o3d

# open3d kdtree 创建
        begin_t = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(db_np)
        pcd_kdtree = o3d.geometry.KDTreeFlann(pcd)
        construction_time_sum += time.time() - begin_t
        

# open3d kdtree knn search
        begin_t = time.time()
        [_, idx_knn, dist_knn] = pcd_kdtree.search_knn_vector_3d(query, k)
        knn_time_sum += time.time() - begin_t

# open3d kdtree radius search
        begin_t = time.time()
        [_, idx_radius, dist_radius] = pcd_kdtree.search_radius_vector_3d(query, radius)        
        radius_time_sum += time.time() - begin_t
        print("Open3d KDTree: build %.3f, knn %.3f, radius %.3f" % (construction_time_sum*1000/iteration_num,
                                                                 knn_time_sum*1000/iteration_num,
                                                                 radius_time_sum*1000/iteration_num))



# 直接暴力搜索
        print("brute force --------------")
        brute_time_sum = 0

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t

        print("brute %.3f" % (brute_time_sum * 1000 / iteration_num))



if __name__ == '__main__':
    main()
