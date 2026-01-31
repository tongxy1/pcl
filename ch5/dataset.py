import torch
import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

def random_rotate_matrix_xyz():
    angles = np.random.rand(3) * 2 * np.pi
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1, 0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,    0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R

def read_pcd_from_file(file):
    np_pts = np.zeros(0)
    with open(file, 'r') as f:
        pts = []
        for line in f:
            one_pt = list(map(float, line[:-1].split(',')))
            pts.append(one_pt[:3])
        np_pts = np.array(pts)
    return np_pts


def read_file_names_from_file(file):
  with open(file, 'r') as f:
    files = []
    for line in f:
      files.append(line.split('\n')[0])
  return files


class PointNetDataset(Dataset):
  def __init__(self, root_dir, train):
    super(PointNetDataset, self).__init__()

    self._train = train
    self._classes = []

    self._features = []
    self._labels = []

    self.load(root_dir)

  def classes(self):
    return self._classes

  def __len__(self):
    return len(self._features)
  
  def __getitem__(self, idx):
    feature, label = self._features[idx], self._labels[idx]
    
    # random sample 1024 points from feature
    OUT_SIZE = 1024
    # rand_size = np.random.randint(500, 5000)
    # choice = np.random.choice(len(feature), rand_size, replace=False)
    # feature = feature[choice, :]
    
    # if rand_size >= OUT_SIZE:
    #   choice = np.random.choice(rand_size, OUT_SIZE, replace=False)      
    # else:      
    #   extra_choice = np.random.choice( rand_size, OUT_SIZE - rand_size, replace=True)
    #   choice = np.concatenate((np.arange(rand_size), extra_choice), axis=0)      
    
    choice = np.random.choice(len(feature), OUT_SIZE, replace=False)
    
    feature = feature[choice, :]

    # TODO: normalize feature
    center = np.mean(feature, axis=0, keepdims=True)  # (N, 3) -> (1, 3)
    dist = np.max(np.sqrt(np.sum((feature-center) ** 2, axis=1)), axis=0)  # (N,3)-(1,3) -> (N,)-> (N,) -> (1)
    feature = (feature - center)/dist  # (N,3)
    
    # center = np.mean(feature, axis=0, keepdims=True)  # (N, 3) -> (1, 3)
    # vars = np.mean((feature-center) ** 2, axis=0, keepdims=True)  # (N,3)-(1,3)-> (N,3) -> (1,3)
    # feature = (feature - center)/np.sqrt(vars+1e-8)  # (N,3)    

    if self._train == 0:
      # TODO: rotation to feature
      theta = np.random.uniform(0, 2 * np.pi)
      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta),  np.cos(theta), 0],
                                  [0,              0,             1]])
      # rotation_matrix = random_rotate_matrix_xyz()
      
      feature = rotation_matrix @ feature.T  # (3, N)    
      feature = feature.T  # (N, 3)

      # jitter
      feature += np.random.normal(0, 0.02, size=feature.shape)
    
    feature = torch.Tensor(feature.T) # (3,N) <-- (N,3)

    l_lable = [0 for _ in range(len(self._classes))]
    l_lable[self._classes.index(label)] = 1
    label = torch.Tensor(l_lable)

    return feature, label
  
  def load(self, root_dir):
    things = os.listdir(root_dir)
    files = []
    for f in things:
      if self._train == 0:
        if f == 'modelnet40_train.txt':
          files = read_file_names_from_file(root_dir + '/' + f)
      elif self._train == 1:
        if f == 'modelnet40_test.txt':
          files = read_file_names_from_file(root_dir + '/' + f)
      if f == "modelnet40_shape_names.txt":
        self._classes = read_file_names_from_file(root_dir + '/' + f)
    tmp_classes = []
    for file in files:
      num = file.split("_")[-1]
      kind = file.split("_" + num)[0]
      if kind not in tmp_classes:
        tmp_classes.append(kind)
      pcd_file = root_dir + '/' + kind + '/' + file + '.txt'
      np_pts = read_pcd_from_file(pcd_file)
      # print(np_pts.shape) # (10000, 3)
      self._features.append(np_pts)
      self._labels.append(kind)
    if self._train == 0:
      print("There are " + str(len(self._labels)) + " trian files.")
    elif self._train == 1:
      print("There are " + str(len(self._labels)) + " test files.")
      

if __name__ == "__main__":
  train_data = PointNetDataset("dataset/modelnet40_normal_resampled", train=0)
  train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
  cnt = 0
  for pts, label in train_loader:
    # print(pts.shape)
    print(label.shape)
    cnt += 1
    if cnt > 3:
      break