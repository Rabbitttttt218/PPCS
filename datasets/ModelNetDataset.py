import os
import sys
import glob
import h5py
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')

def create_zigzag_paths_3d(points, num_paths=6):
    """
    为3D点云创建zigzag填充路径
    
    参数:
        points: 形状为 [N, 3] 的点云数据, 其中 N 是点的数量
        num_paths: 要生成的路径数量
        
    返回:
        paths: 路径列表，每个路径是一个索引数组
        path_indices: 路径的索引列表
    """
    N = points.shape[0]  # 点的总数
    paths = []
    path_indices = []
    
    # 沿着三个主平面创建zigzag路径
    
    # 1. XY平面的zigzag路径 (从不同的Z层开始)
    num_xy_paths = num_paths // 3 + (num_paths % 3 > 0)
    # 按Z坐标分层
    z_sorted = np.argsort(points[:, 2])
    z_layers = np.array_split(z_sorted, num_xy_paths)
    
    for layer_idx, layer_points in enumerate(z_layers):
        if len(layer_points) == 0:
            continue
            
        # 在XY平面上创建zigzag路径
        layer_points_coords = points[layer_points]
        
        # 按X坐标排序
        x_sorted = layer_points[np.argsort(layer_points_coords[:, 0])]
        
        # 在X分段内按Y坐标交替排序
        x_segments = np.array_split(x_sorted, max(1, min(len(x_sorted) // 20, 10)))
        path = []
        
        for i, segment in enumerate(x_segments):
            if len(segment) == 0:
                continue
                
            segment_coords = points[segment]
            
            # 奇数段逆序排列Y，偶数段正序排列Y
            if i % 2 == 0:
                sorted_segment = segment[np.argsort(segment_coords[:, 1])]
            else:
                sorted_segment = segment[np.argsort(-segment_coords[:, 1])]
                
            path.extend(sorted_segment)
            
        if path:
            # 确保路径包含所有点
            seen_indices = set(path)
            unseen_indices = list(set(range(N)) - seen_indices)
            
            # 添加剩余的点到路径末尾
            if unseen_indices:
                path.extend(unseen_indices)
                
            paths.append(np.array(path))
            path_indices.append(len(paths) - 1)
    
    # 2. XZ平面的zigzag路径 (从不同的Y层开始)
    num_xz_paths = num_paths // 3 + (num_paths % 3 > 1)
    # 按Y坐标分层
    y_sorted = np.argsort(points[:, 1])
    y_layers = np.array_split(y_sorted, num_xz_paths)
    
    for layer_idx, layer_points in enumerate(y_layers):
        if len(layer_points) == 0:
            continue
            
        # 在XZ平面上创建zigzag路径
        layer_points_coords = points[layer_points]
        
        # 按X坐标排序
        x_sorted = layer_points[np.argsort(layer_points_coords[:, 0])]
        
        # 在X分段内按Z坐标交替排序
        x_segments = np.array_split(x_sorted, max(1, min(len(x_sorted) // 20, 10)))
        path = []
        
        for i, segment in enumerate(x_segments):
            if len(segment) == 0:
                continue
                
            segment_coords = points[segment]
            
            # 奇数段逆序排列Z，偶数段正序排列Z
            if i % 2 == 0:
                sorted_segment = segment[np.argsort(segment_coords[:, 2])]
            else:
                sorted_segment = segment[np.argsort(-segment_coords[:, 2])]
                
            path.extend(sorted_segment)
            
        if path:
            # 确保路径包含所有点
            seen_indices = set(path)
            unseen_indices = list(set(range(N)) - seen_indices)
            
            # 添加剩余的点到路径末尾
            if unseen_indices:
                path.extend(unseen_indices)
                
            paths.append(np.array(path))
            path_indices.append(len(paths) - 1)
    
    # 3. YZ平面的zigzag路径 (从不同的X层开始)
    num_yz_paths = num_paths // 3
    # 按X坐标分层
    x_sorted = np.argsort(points[:, 0])
    x_layers = np.array_split(x_sorted, num_yz_paths)
    
    for layer_idx, layer_points in enumerate(x_layers):
        if len(layer_points) == 0:
            continue
            
        # 在YZ平面上创建zigzag路径
        layer_points_coords = points[layer_points]
        
        # 按Y坐标排序
        y_sorted = layer_points[np.argsort(layer_points_coords[:, 1])]
        
        # 在Y分段内按Z坐标交替排序
        y_segments = np.array_split(y_sorted, max(1, min(len(y_sorted) // 20, 10)))
        path = []
        
        for i, segment in enumerate(y_segments):
            if len(segment) == 0:
                continue
                
            segment_coords = points[segment]
            
            # 奇数段逆序排列Z，偶数段正序排列Z
            if i % 2 == 0:
                sorted_segment = segment[np.argsort(segment_coords[:, 2])]
            else:
                sorted_segment = segment[np.argsort(-segment_coords[:, 2])]
                
            path.extend(sorted_segment)
            
        if path:
            # 确保路径包含所有点
            seen_indices = set(path)
            unseen_indices = list(set(range(N)) - seen_indices)
            
            # 添加剩余的点到路径末尾
            if unseen_indices:
                path.extend(unseen_indices)
                
            paths.append(np.array(path))
            path_indices.append(len(paths) - 1)
    
    # 如果没有生成任何路径，创建一个默认路径
    if not paths:
        paths.append(np.arange(N))
        path_indices.append(0)
    
    return paths, path_indices

def load_modelnet_data(partition):
    BASE_DIR = './'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


@DATASETS.register_module()
class ModelNet40SVM(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

@DATASETS.register_module()
class ModelNet(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        
        # 添加zigzag相关参数
        self.use_zigzag = config.use_zigzag
        self.num_paths = config.num_paths
        self.zigzag_indices = config.zigzag_indices
        
        # 添加统计计数器
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {}  # 记录每条路径的使用次数
        
        # 下面是原始的数据加载代码，保持不变
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger = 'ModelNet')

        # 重要：保持原有的保存路径，不添加zigzag后缀
        if self.uniform:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # 下面是原始的数据处理代码，保持不变
        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger = 'ModelNet')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger = 'ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
                    
        # 如果使用zigzag，添加日志
        if self.use_zigzag:
            print_log(f'[DATASET] Using zigzag path ordering with {self.num_paths} paths', logger='ModelNet')

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        # 原始的_get_item方法保持不变
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        # 添加调试代码
        if index == 0:  # 只在第一个样本时打印
            with open('/tmp/zigzag_debug_modelnet.txt', 'w') as f:
                f.write(f"First sample accessed: use_zigzag={self.use_zigzag}, num_paths={self.num_paths}\n")
        
        # 更新总样本计数
        self.total_samples += 1
        
        points, label = self._get_item(index)
        
        # 应用zigzag路径或随机排序
        if self.use_zigzag:
            # 为当前点云生成zigzag路径
            paths, path_indices = create_zigzag_paths_3d(points, num_paths=self.num_paths)
            
            # 确保所有路径包含所有点
            for i in range(len(paths)):
                if len(paths[i]) != len(points):
                    print(f"Warning: Path {i} length ({len(paths[i])}) does not match data length ({len(points)})")
            
            # 确保zigzag_indices中的所有索引都是有效的
            valid_indices = [idx for idx in self.zigzag_indices if idx < len(paths)]
            if not valid_indices:
                valid_indices = [0] if len(paths) > 0 else []
            
            # 记录路径使用信息
            path_info = ""
            
            # 根据训练/测试模式选择路径
            if self.subset == 'train':
                if valid_indices:
                    path_idx = np.random.choice(valid_indices) # 随机选择一条路径
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1
                    

                    # 记录路径使用次数
                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1
                    
                    path_info = f"zigzag_path_{path_idx}"
                else:
                    # 如果没有有效路径，使用原始点顺序
                    current_path = np.arange(len(points))
                    self.original_order_counter += 1
                    path_info = "original_order"
            else:
                # 测试时使用第一条有效路径
                if valid_indices:
                    path_idx = valid_indices[0]
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1
                    
                    # 记录路径使用次数
                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1
                    
                    path_info = f"zigzag_path_{path_idx}"
                else:
                    # 如果没有有效路径，使用原始点顺序
                    current_path = np.arange(len(points))
                    self.original_order_counter += 1
                    path_info = "original_order"
            
            # 应用选择的路径对点云重排序
            current_points = points[current_path].copy()
        else:
            # 使用原有的随机排序
            pt_idxs = np.arange(0, points.shape[0])  # 2048
            if self.subset == 'train':
                np.random.shuffle(pt_idxs)
            current_points = points[pt_idxs].copy()
        
        # 统计信息记录（每100个样本打印一次）
        if self.total_samples % 100 == 0 and self.use_zigzag:
            # 统计输出逻辑...
            with open('/tmp/zigzag_path_stats_modelnet.txt', 'a') as f:
                f.write(f"\n\n========== ZigZag Path Usage Statistics ==========\n")
                # 详细统计记录...
            
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)
