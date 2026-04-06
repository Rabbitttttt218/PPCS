import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

def create_zigzag_paths_3d(points, num_paths=6):
    """
    为3D点云创建zigzag填充路径
    
    参数:
        points: 形状为 [N, 3] 的点云数据, 其中 N 是点的数量
        num_paths: 要生成的路径数量
        
    返回:
        paths: 路径列表，每个路径是一个索引数组，包含所有点的索引
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

    
    return paths, path_indices

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')
        
        # 添加zigzag相关参数
        self.use_zigzag = config.use_zigzag
        self.num_paths = config.num_paths
        self.zigzag_indices = config.zigzag_indices
        
        # 添加统计计数器
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {}  # 记录每条路径的使用次数
        
        # 直接使用print打印配置信息，确保可见
        # print(f"\n\n=============== ZigZag Path Configuration for ShapeNet ===============")
        # print(f"Dataset: {self.subset}")
        # print(f"use_zigzag: {self.use_zigzag}")
        # print(f"num_paths: {self.num_paths}")
        # print(f"zigzag_indices: {self.zigzag_indices}")
        # print(f"======================================================================\n\n")
        
        # 创建调试文件
        # with open('/tmp/zigzag_debug.txt', 'w') as f:
            # f.write(f"ShapeNet Dataset Initialization\n")
            # f.write(f"use_zigzag: {self.use_zigzag}\n")
            # f.write(f"num_paths: {self.num_paths}\n")
            # f.write(f"zigzag_indices: {self.zigzag_indices}\n")

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                  test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')
        
        # 保存随机采样的排列
        self.permutation = np.arange(self.npoints)
        
        # 如果使用zigzag，可以预先计算并缓存路径
        if self.use_zigzag:
            print_log(f'[DATASET] Using zigzag path ordering with {self.num_paths} paths', logger='ShapeNet-55')

    # def pc_norm(self, pc):
    #     """ pc: NxC, return NxC """
    #     centroid = np.mean(pc, axis=0)
    #     pc = pc - centroid
    #     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    #     pc = pc / m
    #     return pc


    
    # # 保留
    def pc_norm(self, pc):
        """ 归一化点云到单位球内 """
        centroid = np.mean(pc, axis=0)  # 计算质心
        pc = pc - centroid  # 中心化
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 计算最大距离
        m = m if m != 0 else 1.0  # 防止除零
        pc = pc / m  # 归一化到单位球
        return pc


    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def __getitem__(self, idx):
        # 添加调试代码
        # if idx == 0:  # 只在第一个样本时打印
            # with open('/tmp/zigzag_debug.txt', 'a') as f:
                # f.write(f"First sample accessed: use_zigzag={self.use_zigzag}, num_paths={self.num_paths}\n")
        
        # 更新总样本计数
        self.total_samples += 1
        
        sample = self.file_list[idx]
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        
        # 首先使用原有的random_sample方法采样1024个点
        data = self.random_sample(data, self.sample_points_num)
        
        # 归一化点云
        data = self.pc_norm(data)
        
        # 如果启用了zigzag，对已采样的1024个点应用zigzag重排序
        if self.use_zigzag:
            # 为当前点云生成zigzag路径
            paths, path_indices = create_zigzag_paths_3d(data, num_paths=self.num_paths)
            
            # 确保所有路径包含所有点
            for i in range(len(paths)):
                if len(paths[i]) != len(data):
                    print(f"Warning: Path {i} length ({len(paths[i])}) does not match data length ({len(data)})")
            
            # 确保zigzag_indices中的所有索引都是有效的
            valid_indices = [idx for idx in self.zigzag_indices if idx < len(paths)]

            
            # 记录路径使用信息
            path_info = ""
            
            # 始终从6条路径中选择
            if valid_indices:
                path_idx = np.random.choice(valid_indices)
                current_path = paths[path_idx]
                self.zigzag_path_counter += 1
                
                # 记录路径使用次数
                if path_idx not in self.path_usage:
                    self.path_usage[path_idx] = 0
                self.path_usage[path_idx] += 1
                
                path_info = f"zigzag_path_{path_idx}"
            else:
                # 如果没有有效路径，使用原始点顺序
                current_path = np.arange(len(data))
                self.original_order_counter += 1
                path_info = "original_order"
            
            # 应用选择的路径对点云重排序
            data = data[current_path]
            
            # 每100个样本打印一次统计信息
            if self.total_samples % 100 == 0:
                zigzag_percent = self.zigzag_path_counter / self.total_samples * 100
                original_percent = self.original_order_counter / self.total_samples * 100
                
                # print(f"\n\n========== ZigZag Path Usage Statistics ==========")
                # print(f"Total samples: {self.total_samples}")
                # print(f"ZigZag paths: {self.zigzag_path_counter} ({zigzag_percent:.2f}%)")
                # print(f"Original order: {self.original_order_counter} ({original_percent:.2f}%)")
                # print(f"Current sample ({idx}): Using {path_info}")
                # print(f"Path length: {len(current_path)}, Data length: {len(data)}")
                
                # 打印每条路径的使用情况
                # print(f"Path usage details:")
                for p_idx, count in sorted(self.path_usage.items()):
                    path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
                    # print(f"  Path {p_idx}: {count} ({path_percent:.2f}%)")
                
                # 检查前10个点的索引是否有变化
                if len(current_path) > 10:
                    original_indices = list(range(10))
                    current_indices = current_path[:10].tolist() if isinstance(current_path, np.ndarray) else current_path[:10]
                    # print(f"First 10 indices - Original: {original_indices}")
                    # print(f"First 10 indices - Current: {current_indices}")
                    # print(f"Indices are different: {original_indices != current_indices}")
                
                # print(f"================================================\n\n")
                
                # 同时写入文件
                # with open('/tmp/zigzag_path_stats.txt', 'a') as f:
                    # f.write(f"\n\n========== ZigZag Path Usage Statistics ==========\n")
                    # f.write(f"Total samples: {self.total_samples}\n")
                    # f.write(f"ZigZag paths: {self.zigzag_path_counter} ({zigzag_percent:.2f}%)\n")
                    # f.write(f"Original order: {self.original_order_counter} ({original_percent:.2f}%)\n")
                    # f.write(f"Current sample ({idx}): Using {path_info}\n")
                    # f.write(f"Path length: {len(current_path)}, Data length: {len(data)}\n")
                    
                    # 打印每条路径的使用情况
                    # f.write(f"Path usage details:\n")
                    # for p_idx, count in sorted(self.path_usage.items()):
                        # path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
                        # f.write(f"  Path {p_idx}: {count} ({path_percent:.2f}%)\n")
        
        # 转换为PyTorch张量
        data = torch.from_numpy(data).float()
        
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)
        