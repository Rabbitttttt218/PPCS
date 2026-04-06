import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *

import cv2
import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",  # plane
        "04379243",  # table
        "03790512",  # motorbike
        "03948459",  # pistol
        "03642806",  # laptop
        "03467517",  # guitar
        "03261776",  # earphone
        "03001627",  # chair
        "02958343",  # car
        "04090263",  # rifle
        "03759954",  # microphone
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if taxonomy_ids[0] not in useful_cate:
                continue
            if taxonomy_ids[0] == "02691156":
                a, b = 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45
            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            else:
                a, b = 0, 0

            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers = base_model(points, vis=True)
            final_image = []
            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points, a, b)
            final_image.append(points[150:650, 150:675, :])

            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)

            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_ptcloud_img(vis_points, a, b)

            final_image.append(vis_points[150:650, 150:675, :])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points, a, b)
            final_image.append(dense_points[150:650, 150:675, :])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)

            if idx > 1500:
                break

        return

# # visualization2-ModelNet40
# def test(base_model, test_dataloader, args, config, logger=None):
#     base_model.eval()  # set model to eval mode
#     target = '/home/dlss107552304043/PointMamba/figure/vis'
#     useful_cate = [
#         "airplane",
#         "bathtub",
#         "bed",
#         "bench",
#         "bookshelf",
#         "bottle",
#         "bowl",
#         "car",
#         "chair",
#         "cone",
#         "cup",
#         "curtain",
#         "desk",
#         "door",
#         "dresser",
#         "flower_pot",
#         "glass_box",
#         "guitar",
#         "keyboard",
#         "lamp",
#         "laptop",
#         "mantel",
#         "monitor",
#         "night_stand",
#         "person",
#         "piano",
#         "plant",
#         "radio",
#         "range_hood",
#         "sink",
#         "sofa",
#         "stairs",
#         "stool",
#         "table",
#         "tent",
#         "toilet",
#         "tv_stand",
#         "vase",
#         "wardrobe",
#         "xbox"
#     ]
#     with torch.no_grad():
#         for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
#             if taxonomy_ids[0] not in useful_cate:
#                 continue
#             # 设置视角参数
#             if taxonomy_ids[0] == "airplane":
#                 a, b = 90, 135
#             elif taxonomy_ids[0] == "bathtub":
#                 a, b = 30, 30
#             elif taxonomy_ids[0] == "bed":
#                 a, b = 30, -45
#             else:
#                 a, b = 0, 0

#             dataset_name = config.dataset.test._base_.NAME
#             print_log(f'Dataset name: {dataset_name}', logger=logger)  # 添加在这里
#             if dataset_name == 'ModelNet40':
#                 points = data.cuda()
#             else:
#                 raise NotImplementedError(f'Train phase do not support {dataset_name}')

#             dense_points, vis_points, centers = base_model(points, vis=True)
#             final_image = []
#             data_path = f'/home/dlss107552304043/PointMamba/figure/vis/{taxonomy_ids[0]}_{idx}'
#             print_log(f'Saving visualization to {data_path}', logger=logger)  # 添加路径日志
#             if not os.path.exists(data_path):
#                 os.makedirs(data_path)

#             points = points.squeeze().detach().cpu().numpy()
#             np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
#             points = misc.get_ptcloud_img(points, a, b)
#             final_image.append(points[150:650, 150:675, :])

#             vis_points = vis_points.squeeze().detach().cpu().numpy()
#             np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
#             vis_points = misc.get_ptcloud_img(vis_points, a, b)
#             final_image.append(vis_points[150:650, 150:675, :])

#             dense_points = dense_points.squeeze().detach().cpu().numpy()
#             np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
#             dense_points = misc.get_ptcloud_img(dense_points, a, b)
#             final_image.append(dense_points[150:650, 150:675, :])

#             img = np.concatenate(final_image, axis=1)
#             img = np.clip(img, 0, 255).astype(np.uint8)  # 确保图像值在 [0, 255] 范围内
#             img_path = os.path.join(data_path, f'plot.jpg')
#             cv2.imwrite(img_path, img)

#             if idx > 1500:
#                 break

#         return


# # 可视化保存重建后的点云数据以及掩码点的点云数据
# import torch
# import os
# import numpy as np
# import cv2
# from tools import builder
# from utils import misc
# from utils.logger import *

# def test_net(args, config):
#     logger = get_logger(args.log_name)
#     print_log('Tester start ... ', logger=logger)

#     _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
#     base_model = builder.model_builder(config.model)

#     # 加载预训练模型
#     builder.load_model(base_model, args.ckpts, logger=logger)

#     if args.use_gpu:
#         base_model.to(args.local_rank)

#     #  DDP
#     if args.distributed:
#         raise NotImplementedError()

#     test(base_model, test_dataloader, args, config, logger=logger)

# def test(base_model, test_dataloader, args, config, logger=None):
#     """测试模型，并保存重建点云和掩码点云数据"""
#     base_model.eval()
    
#     # 设置存储路径
#     # save_dir = "/home/dlss107552304043/PointMamba"
#     save_dir = "/home/dlss107552304043/PointVisualizaiton-main/example"
#     os.makedirs(save_dir, exist_ok=True)

#     useful_cate = [
#         "02691156",  # plane
#         "04379243",  # table
#         "03790512",  # motorbike
#         "03948459",  # pistol
#         "03642806",  # laptop
#         "03467517",  # guitar
#         "03261776",  # earphone
#         "03001627",  # chair
#         "02958343",  # car
#         "04090263",  # rifle
#         "03759954",  # microphone
#     ]
    
#     with torch.no_grad():
#         for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
#             if taxonomy_ids[0] not in useful_cate:
#                 continue

#             dataset_name = config.dataset.test._base_.NAME
#             if dataset_name == 'ShapeNet':
#                 points = data.cuda()
#             else:
#                 raise NotImplementedError(f'Train phase do not support {dataset_name}')

#             # 获取模型输出
#             dense_points, vis_points, centers = base_model(points, vis=True)

#             # ✅ **保存重建点云和掩码点云**
#             rebuild_save_path = os.path.join(save_dir, f"rebuild_points_{idx}.npy")
#             masked_save_path = os.path.join(save_dir, f"masked_points_{idx}.npy")

#             np.save(rebuild_save_path, dense_points.cpu().numpy())
#             np.save(masked_save_path, vis_points.cpu().numpy())

#             print(f"✅ {idx}: 重建点云数据已保存到 {rebuild_save_path}")
#             print(f"✅ {idx}: 掩码点云数据已保存到 {masked_save_path}")

#             # =======================
#             # 🔹 仍然保留原有的可视化保存功能
#             # =======================
#             # data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
#             data_path = f'/home/dlss107552304043/PointMamba/figure/vis/{taxonomy_ids[0]}_{idx}'
#             os.makedirs(data_path, exist_ok=True)

#             points = points.squeeze().detach().cpu().numpy()
#             np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
#             points = misc.get_ptcloud_img(points, 0, 0)

#             vis_points = vis_points.squeeze().detach().cpu().numpy()
#             np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
#             vis_points = misc.get_ptcloud_img(vis_points, 0, 0)

#             dense_points = dense_points.squeeze().detach().cpu().numpy()
#             np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
#             dense_points = misc.get_ptcloud_img(dense_points, 0, 0)

#             # 组合可视化结果并保存
#             final_image = [points[150:650, 150:675, :], vis_points[150:650, 150:675, :], dense_points[150:650, 150:675, :]]
#             img = np.concatenate(final_image, axis=1)
#             cv2.imwrite(os.path.join(data_path, 'plot.jpg'), img)

#             if idx > 50:  # 限制测试数量，避免保存过多数据
#                 break

#     return
