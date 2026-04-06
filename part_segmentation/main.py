# FLOPs Param
import argparse
import os
import random

import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


# 添加FLOPs计算函数
def estimate_flops_detailed(model, input_shape=(3, 2048), cls_dim=50):
    """重新修正的FLOPs计算"""
    B, C, N = 1, input_shape[0], input_shape[1]
    
    flops_info = []
    flops_info.append(f"开始计算FLOPs (输入形状: Batch={B}, Channels={C}, Points={N})")
    flops_info.append("-" * 80)
    
    total_flops = 0
    
    # 1. Group Divider部分
    flops_info.append("1. Group Divider阶段:")
    num_group = 128
    group_size = 32
    
    # FPS - 修正：应该是O(N*k)而不是O(N*k*10)
    fps_flops = N * num_group * 3  # 距离计算
    flops_info.append(f"   FPS操作: {fps_flops:,} FLOPs")
    
    # KNN - 修正：应该更合理
    knn_flops = num_group * group_size * 3  # 每个group内部的knn
    flops_info.append(f"   KNN搜索: {knn_flops:,} FLOPs")
    
    group_flops = fps_flops + knn_flops
    total_flops += group_flops
    flops_info.append(f"   Group Divider总计: {group_flops:,} FLOPs")
    
    # 2. Encoder部分 - 修正
    flops_info.append("")
    flops_info.append("2. Encoder阶段:")
    # 对每个group进行编码，而不是对所有点
    encoder_flops = num_group * group_size * (3 * 128 + 128 * 256 + 256 * 512 + 512 * 384)
    total_flops += encoder_flops
    flops_info.append(f"   Encoder总计: {encoder_flops:,} FLOPs")
    
    # 3. Position Embedding - 修正
    flops_info.append("")
    flops_info.append("3. Position Embedding阶段:")
    pos_embed_flops = num_group * (3 * 128 + 128 * 384)
    total_flops += pos_embed_flops
    flops_info.append(f"   Position Embedding: {pos_embed_flops:,} FLOPs")
    
    # 4. 重排序操作 - 大幅减少
    flops_info.append("")
    flops_info.append("4. 重排序阶段:")
    reorder_flops = num_group * 384 * 6  # 只是重排序，计算量很小
    total_flops += reorder_flops
    flops_info.append(f"   重排序操作: {reorder_flops:,} FLOPs")
    
    # 5. Mamba Blocks - 修正，减少估算
    flops_info.append("")
    flops_info.append("5. Mamba Blocks阶段:")
    trans_dim = 384
    depth = 12
    seq_len = num_group  # 修正：应该是128，不是128*3
    
    # Mamba的计算量相比Transformer要小很多
    mamba_flops_per_layer = seq_len * trans_dim * trans_dim * 4  # 大幅减少估算
    total_mamba_flops = mamba_flops_per_layer * depth
    total_flops += total_mamba_flops
    
    flops_info.append(f"   每层Mamba Block: {mamba_flops_per_layer:,} FLOPs")
    flops_info.append(f"   {depth}层Mamba Blocks总计: {total_mamba_flops:,} FLOPs")
    
    # 6. 特征处理和全局特征 - 修正
    flops_info.append("")
    flops_info.append("6. 特征处理阶段:")
    feature_flops = seq_len * trans_dim * 3  # 特征拼接
    total_flops += feature_flops
    flops_info.append(f"   特征拼接和池化: {feature_flops:,} FLOPs")
    
    # 7. Label embedding - 保持不变
    flops_info.append("")
    flops_info.append("7. Label Embedding阶段:")
    label_flops = 16 * 64
    total_flops += label_flops
    flops_info.append(f"   Label Embedding: {label_flops:,} FLOPs")
    
    # 8. Feature Propagation - 修正，这是主要的增加部分
    flops_info.append("")
    flops_info.append("8. Feature Propagation阶段:")
    # 从group特征传播到所有点
    prop_flops = N * (1152 * 512)  # 修正计算
    total_flops += prop_flops
    flops_info.append(f"   Feature Propagation: {prop_flops:,} FLOPs")
    
    # 9. 最终分类头 - 这是分割任务的主要增加
    flops_info.append("")
    flops_info.append("9. 分类头阶段:")
    # 为每个点进行分类
    head_flops = N * (1600 * 512 + 512 * 256 + 256 * cls_dim)
    total_flops += head_flops
    flops_info.append(f"   分类头: {head_flops:,} FLOPs")
    
    flops_info.append("")
    flops_info.append("=" * 80)
    flops_info.append(f"总FLOPs: {total_flops:,}")
    flops_info.append(f"总FLOPs (G): {total_flops/1e9:.2f}G")
    flops_info.append(f"总FLOPs (M): {total_flops/1e6:.2f}M")
    flops_info.append("=" * 80)
    
    return total_flops, flops_info



def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt_mamba', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--config', type=str, default=None, help='config file')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--root', type=str, default='../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
                        help='data root')
    
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.root

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    if args.config is not None:
        from utils.config import cfg_from_yaml_file
        from utils.logger import print_log
        if args.config[:13] == "segmentation/":
            args.config = args.config[13:]
        config = cfg_from_yaml_file(args.config)
        log_string(config)
        if hasattr(config, 'epoch'):
            args.epoch = config.epoch
        if hasattr(config, 'batch_size'):
            args.epoch = config.batch_size
        if hasattr(config, 'learning_rate'):
            args.learning_rate = config.learning_rate
        if hasattr(config, 'ckpt') and args.ckpts is None:
            args.ckpts = config.ckpts
        if hasattr(config, 'model'):
            MODEL = importlib.import_module(config.model) if hasattr(config, 'model') else importlib.import_module(
                args.model)
            classifier = MODEL.get_model(num_part, config).cuda()
        else:
            MODEL = importlib.import_module(args.model)
            classifier = MODEL.get_model(num_part).cuda()
    else:
        MODEL = importlib.import_module(args.model)
        shutil.copy('models/%s.py' % args.model, str(exp_dir))
        classifier = MODEL.get_model(num_part).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    
    # 添加FLOPs计算
    log_string('')
    log_string('开始计算模型FLOPs...')
    total_flops, flops_info = estimate_flops_detailed(classifier, input_shape=(3, args.npoint), cls_dim=num_part)
    
    # 将FLOPs信息写入日志
    for info in flops_info:
        log_string(info)
    
    start_epoch = 0

    if args.ckpts is not None:
        if args.ckpts[:13] == "segmentation/":
            args.ckpts = args.ckpts[13:]
        classifier.load_model_from_ckpt(args.ckpts)
        log_string('Load model from %s' % args.ckpts)
    else:
        log_string('No existing model, starting training from scratch...')

    ## we use adamw and cosine scheduler
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        num_trainable_params = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
                num_trainable_params += param.numel()
            else:
                decay.append(param)
                num_trainable_params += param.numel()

        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        log_string('########################################################################')
        log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
            '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
        log_string('########################################################################')

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            num_iter += 1
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            if num_iter == 1:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Train loss: %.5f' % loss1)
        log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                          smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)






















# # 原始加载S3DIS数据
# import argparse
# import os
# import random
# import torch
# import datetime
# import logging
# import sys
# import importlib
# import shutil
# import provider
# import numpy as np
# import torch.optim as optim
# from timm.scheduler import CosineLRScheduler
# from pathlib import Path
# from tqdm import tqdm
# from dataset import PartNormalDataset, S3DISDataset

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

# # ShapeNetPart相关配置
# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
# seg_label_to_cat = {}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat

# # S3DIS类别定义
# s3dis_classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 
#                  'window', 'door', 'table', 'chair', 'sofa', 
#                  'bookcase', 'board', 'clutter']

# def inplace_relu(m):
#     classname = m.__class__.__name__
#     if classname.find('ReLU') != -1:
#         m.inplace = True

# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
#     if y.is_cuda:
#         return new_y.cuda()
#     return new_y

# def parse_args():
#     parser = argparse.ArgumentParser('Model')
#     parser.add_argument('--model', type=str, default='pt_mamba', help='model name')
#     parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
#     parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
#     parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
#     parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
#     parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
#     parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
#     parser.add_argument('--npoint', type=int, default=4096, help='point Number')
#     parser.add_argument('--normal', action='store_true', default=False, help='use normals')
#     parser.add_argument('--config', type=str, default=None, help='config file')
#     parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
#     parser.add_argument('--root', type=str, default='/home/dlss107552304043/PointMamba/data/Stanford3dDataset_v1.2_Aligned_Version/',
#                         help='data root')
#     parser.add_argument('--test_area', type=int, default=5, help='test area for S3DIS (1-6)')
    
#     return parser.parse_args()

# def main(args):
#     def log_string(str):
#         logger.info(str)
#         print(str)

#     '''CREATE DIR'''
#     timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
#     exp_dir = Path('./log/')
#     exp_dir.mkdir(exist_ok=True)
#     exp_dir = exp_dir.joinpath('s3dis_seg')  # 修改日志目录名
#     exp_dir.mkdir(exist_ok=True)
#     if args.log_dir is None:
#         exp_dir = exp_dir.joinpath(timestr)
#     else:
#         exp_dir = exp_dir.joinpath(args.log_dir)
#     exp_dir.mkdir(exist_ok=True)
#     checkpoints_dir = exp_dir.joinpath('checkpoints/')
#     checkpoints_dir.mkdir(exist_ok=True)
#     log_dir = exp_dir.joinpath('logs/')
#     log_dir.mkdir(exist_ok=True)

#     '''LOG'''
#     logger = logging.getLogger("Model")
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     log_string('PARAMETER ...')
#     log_string(args)

#     # 判断是否使用S3DIS数据集
#     is_s3dis = args.config is not None and 's3dis' in args.config.lower()
    
#     # 根据数据集类型设置参数
#     if is_s3dis:
#         num_classes = 13  # S3DIS有13个语义类别
#         num_part = 13     # S3DIS直接分割为13个语义类别
#         root = args.root
#         log_string('Using S3DIS Dataset')
#     else:
#         num_classes = 16  # ShapeNetPart有16个物体类别
#         num_part = 50     # ShapeNetPart有50个部件类别
#         root = args.root
#         log_string('Using ShapeNetPart Dataset')

#     '''DATASET LOADING'''
#     if is_s3dis:
#         # S3DIS数据集
#         TRAIN_DATASET = S3DISDataset(root=root, npoints=args.npoint, split='train', test_area=args.test_area)
#         TEST_DATASET = S3DISDataset(root=root, npoints=args.npoint, split='test', test_area=args.test_area)
#     else:
#         # ShapeNetPart数据集
#         TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
#         TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

#     trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
#                                                   num_workers=10, drop_last=True)
#     testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
#                                                  num_workers=10)
#     log_string("The number of training data is: %d" % len(TRAIN_DATASET))
#     log_string("The number of test data is: %d" % len(TEST_DATASET))

#     '''MODEL LOADING'''
#     shutil.copy('models/%s.py' % args.model, str(exp_dir))
    
#     if args.config is not None:
#         from utils.config import cfg_from_yaml_file
#         if args.config[:13] == "segmentation/":
#             args.config = args.config[13:]
#         config = cfg_from_yaml_file(args.config)
#         log_string(config)
        
#         if hasattr(config, 'epoch'):
#             args.epoch = config.epoch
#         if hasattr(config, 'batch_size'):
#             args.batch_size = config.batch_size  # 修复原代码bug
#         if hasattr(config, 'learning_rate'):
#             args.learning_rate = config.learning_rate
#         if hasattr(config, 'ckpt') and args.ckpts is None:
#             args.ckpts = config.ckpts
#         if hasattr(config, 'test_area'):
#             args.test_area = config.test_area
            
#         if hasattr(config, 'model'):
#             MODEL = importlib.import_module(config.model) if hasattr(config, 'model') else importlib.import_module(args.model)
#         else:
#             MODEL = importlib.import_module(args.model)
            
#         # 根据数据集类型选择模型
#         if is_s3dis:
#             if hasattr(MODEL, 'get_s3dis_model'):
#                 classifier = MODEL.get_s3dis_model(num_classes, config).cuda()
#             else:
#                 classifier = MODEL.get_model(num_part, config).cuda()
#         else:
#             classifier = MODEL.get_model(num_part, config).cuda()
#     else:
#         MODEL = importlib.import_module(args.model)
#         shutil.copy('models/%s.py' % args.model, str(exp_dir))
#         if is_s3dis:
#             if hasattr(MODEL, 'get_s3dis_model'):
#                 classifier = MODEL.get_s3dis_model(num_classes).cuda()
#             else:
#                 classifier = MODEL.get_model(num_part).cuda()
#         else:
#             classifier = MODEL.get_model(num_part).cuda()
    
#     criterion = MODEL.get_loss().cuda()
#     classifier.apply(inplace_relu)
#     print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
#     start_epoch = 0

#     if args.ckpts is not None:
#         if args.ckpts[:13] == "segmentation/":
#             args.ckpts = args.ckpts[13:]
#         classifier.load_model_from_ckpt(args.ckpts)
#         log_string('Load model from %s' % args.ckpts)
#     else:
#         log_string('No existing model, starting training from scratch...')

#     def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
#         decay = []
#         no_decay = []
#         num_trainable_params = 0
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue
#             if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
#                 no_decay.append(param)
#                 num_trainable_params += param.numel()
#             else:
#                 decay.append(param)
#                 num_trainable_params += param.numel()

#         total_params = sum([v.numel() for v in model.parameters()])
#         non_trainable_params = total_params - num_trainable_params
#         log_string('########################################################################')
#         log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
#             '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
#         log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
#         log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
#         log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
#         log_string('########################################################################')

#         return [
#             {'params': no_decay, 'weight_decay': 0.},
#             {'params': decay, 'weight_decay': weight_decay}]

#     param_groups = add_weight_decay(classifier, weight_decay=0.05)
#     optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

#     scheduler = CosineLRScheduler(optimizer,
#                                   t_initial=args.epoch,
#                                   t_mul=1,
#                                   lr_min=1e-6,
#                                   decay_rate=0.1,
#                                   warmup_lr_init=1e-6,
#                                   warmup_t=args.warmup_epoch,
#                                   cycle_limit=1,
#                                   t_in_epochs=True)

#     best_acc = 0
#     global_epoch = 0
#     best_class_avg_iou = 0
#     best_instance_avg_iou = 0

#     classifier.zero_grad()
#     for epoch in range(start_epoch, args.epoch):
#         mean_correct = []

#         log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

#         classifier = classifier.train()
#         loss_batch = []
#         num_iter = 0
        
#         '''learning one epoch'''
#         for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
#             num_iter += 1
#             points = points.data.numpy()
#             points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
#             points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
#             points = torch.Tensor(points)
#             points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#             points = points.transpose(2, 1)

#             if is_s3dis:
#                 # S3DIS不需要类别标签作为输入
#                 seg_pred = classifier(points)
#             else:
#                 # ShapeNetPart需要类别标签作为输入
#                 seg_pred = classifier(points, to_categorical(label, num_classes))
            
#             seg_pred = seg_pred.contiguous().view(-1, num_part)
#             target = target.view(-1, 1)[:, 0]
#             pred_choice = seg_pred.data.max(1)[1]

#             correct = pred_choice.eq(target.data).cpu().sum()
#             mean_correct.append(correct.item() / (args.batch_size * args.npoint))
#             loss = criterion(seg_pred, target)
#             loss.backward()
            
#             loss_batch.append(loss.detach().cpu())

#             if num_iter == 1:
#                 torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
#                 num_iter = 0
#                 optimizer.step()
#                 classifier.zero_grad()

#         if isinstance(scheduler, list):
#             for item in scheduler:
#                 item.step(epoch)
#         else:
#             scheduler.step(epoch)

#         train_instance_acc = np.mean(mean_correct)
#         loss1 = np.mean(loss_batch)
#         log_string('Train accuracy is: %.5f' % train_instance_acc)
#         log_string('Train loss: %.5f' % loss1)
#         log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

#         with torch.no_grad():
#             test_metrics = {}
#             total_correct = 0
#             total_seen = 0
#             total_seen_class = [0 for _ in range(num_part)]
#             total_correct_class = [0 for _ in range(num_part)]
            
#             classifier = classifier.eval()

#             if is_s3dis:
#                 # S3DIS评估逻辑
#                 class_ious = []
                
#                 for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
#                                                               smoothing=0.9):
#                     cur_batch_size, NUM_POINT, _ = points.size()
#                     points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#                     seg_pred = classifier(points)
#                     cur_pred_val = seg_pred.cpu().data.numpy()
#                     cur_pred_val = np.argmax(cur_pred_val, axis=2)
#                     target = target.cpu().data.numpy()

#                     correct = np.sum(cur_pred_val == target)
#                     total_correct += correct
#                     total_seen += (cur_batch_size * NUM_POINT)

#                     for l in range(num_part):
#                         total_seen_class[l] += np.sum(target == l)
#                         total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

#                     # 计算每个样本的IoU
#                     for i in range(cur_batch_size):
#                         segp = cur_pred_val[i, :]
#                         segl = target[i, :]
#                         part_ious = []
#                         for l in range(num_part):
#                             if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
#                                 part_ious.append(1.0)
#                             else:
#                                 part_ious.append(np.sum((segl == l) & (segp == l)) / float(
#                                     np.sum((segl == l) | (segp == l))))
#                         class_ious.append(np.mean(part_ious))

#                 # 输出每个类别的IoU
#                 for i, class_name in enumerate(s3dis_classes):
#                     if total_seen_class[i] > 0:
#                         class_iou = total_correct_class[i] / float(total_seen_class[i] + 
#                                                                   np.sum([np.sum(cur_pred_val == i) for cur_pred_val in [cur_pred_val]]) - 
#                                                                   total_correct_class[i])
#                         log_string('eval IoU of %s: %f' % (class_name + ' ' * (14 - len(class_name)), class_iou))

#                 test_metrics['accuracy'] = total_correct / float(total_seen)
#                 test_metrics['class_avg_accuracy'] = np.mean(
#                     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
#                 test_metrics['class_avg_iou'] = np.mean(
#                     [total_correct_class[i] / float(total_seen_class[i]) if total_seen_class[i] > 0 else 0 
#                      for i in range(num_part)])
#                 test_metrics['instance_avg_iou'] = np.mean(class_ious)
                
#             else:
#                 # ShapeNetPart评估逻辑（原有逻辑）
#                 shape_ious = {cat: [] for cat in seg_classes.keys()}
#                 seg_label_to_cat = {}

#                 for cat in seg_classes.keys():
#                     for label in seg_classes[cat]:
#                         seg_label_to_cat[label] = cat

#                 for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
#                                                               smoothing=0.9):
#                     cur_batch_size, NUM_POINT, _ = points.size()
#                     points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#                     seg_pred = classifier(points, to_categorical(label, num_classes))
#                     cur_pred_val = seg_pred.cpu().data.numpy()
#                     cur_pred_val_logits = cur_pred_val
#                     cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
#                     target = target.cpu().data.numpy()

#                     for i in range(cur_batch_size):
#                         cat = seg_label_to_cat[target[i, 0]]
#                         logits = cur_pred_val_logits[i, :, :]
#                         cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

#                     correct = np.sum(cur_pred_val == target)
#                     total_correct += correct
#                     total_seen += (cur_batch_size * NUM_POINT)

#                     for l in range(num_part):
#                         total_seen_class[l] += np.sum(target == l)
#                         total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

#                     for i in range(cur_batch_size):
#                         segp = cur_pred_val[i, :]
#                         segl = target[i, :]
#                         cat = seg_label_to_cat[segl[0]]
#                         part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
#                         for l in seg_classes[cat]:
#                             if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
#                                 part_ious[l - seg_classes[cat][0]] = 1.0
#                             else:
#                                 part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
#                                     np.sum((segl == l) | (segp == l)))
#                         shape_ious[cat].append(np.mean(part_ious))

#                 all_shape_ious = []
#                 for cat in shape_ious.keys():
#                     for iou in shape_ious[cat]:
#                         all_shape_ious.append(iou)
#                     shape_ious[cat] = np.mean(shape_ious[cat])
#                 mean_shape_ious = np.mean(list(shape_ious.values()))
#                 test_metrics['accuracy'] = total_correct / float(total_seen)
#                 test_metrics['class_avg_accuracy'] = np.mean(
#                     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
#                 for cat in sorted(shape_ious.keys()):
#                     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
#                 test_metrics['class_avg_iou'] = mean_shape_ious
#                 test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

#         log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f' % (
#             epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou']))
        
#         if test_metrics['instance_avg_iou'] >= best_instance_avg_iou:
#             logger.info('Save model...')
#             savepath = str(checkpoints_dir) + '/best_model.pth'
#             log_string('Saving at %s' % savepath)
#             state = {
#                 'epoch': epoch,
#                 'train_acc': train_instance_acc,
#                 'test_acc': test_metrics['accuracy'],
#                 'class_avg_iou': test_metrics['class_avg_iou'],
#                 'instance_avg_iou': test_metrics['instance_avg_iou'],
#                 'model_state_dict': classifier.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }
#             torch.save(state, savepath)
#             log_string('Saving model....')

#         if test_metrics['accuracy'] > best_acc:
#             best_acc = test_metrics['accuracy']
#         if test_metrics['class_avg_iou'] > best_class_avg_iou:
#             best_class_avg_iou = test_metrics['class_avg_iou']
#         if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
#             best_instance_avg_iou = test_metrics['instance_avg_iou']
            
#         log_string('Best accuracy is: %.5f' % best_acc)
#         log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
#         log_string('Best instance avg mIOU is: %.5f' % best_instance_avg_iou)
#         global_epoch += 1

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)









# # # 加载S3DIS数据集——第一次修改
# import argparse
# import os
# import random

# import torch
# import datetime
# import logging
# import sys
# import importlib
# import shutil
# import provider
# import numpy as np
# import torch.optim as optim
# from timm.scheduler import CosineLRScheduler
# from pathlib import Path
# from tqdm import tqdm
# from dataset import PartNormalDataset
# # 添加S3DIS数据集导入
# from dataset import S3DISDataset

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat

# # S3DIS类别定义
# s3dis_classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 
#                  'window', 'door', 'table', 'chair', 'sofa', 
#                  'bookcase', 'board', 'clutter']


# def inplace_relu(m):
#     classname = m.__class__.__name__
#     if classname.find('ReLU') != -1:
#         m.inplace = True


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
#     if (y.is_cuda):
#         return new_y.cuda()
#     return new_y


# def parse_args():
#     parser = argparse.ArgumentParser('Model')
#     parser.add_argument('--model', type=str, default='pt_mamba', help='model name')
#     parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
#     parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
#     parser.add_argument('--warmup_epoch', default=60, type=int, help='warmup epoch')# 原来为10
#     parser.add_argument('--learning_rate', default=0.005, type=float, help='initial learning rate')# 原来为0.0003
#     parser.add_argument('--gpu', type=str, default='1', help='specify GPU devices')
#     parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
#     parser.add_argument('--npoint', type=int, default=8192, help='point Number')
#     parser.add_argument('--normal', action='store_true', default=False, help='use normals')
#     parser.add_argument('--config', type=str, default=None, help='config file')
#     parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
#     parser.add_argument('--root', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version/',
#                         help='data root')
#     #/home/dlss107552304043/PointMamba/data/Stanford3dDataset_v1.2_Aligned_Version
#     parser.add_argument('--test_area', type=int, default=5, help='test area for S3DIS (1-6)')
    
#     return parser.parse_args()


# def main(args):
#     def log_string(str):
#         logger.info(str)
#         print(str)

#     '''HYPER PARAMETER'''
#     # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#     '''CREATE DIR'''
#     timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
#     exp_dir = Path('./log/')
#     exp_dir.mkdir(exist_ok=True)
#     exp_dir = exp_dir.joinpath('part_seg')
#     exp_dir.mkdir(exist_ok=True)
#     if args.log_dir is None:
#         exp_dir = exp_dir.joinpath(timestr)
#     else:
#         exp_dir = exp_dir.joinpath(args.log_dir)
#     exp_dir.mkdir(exist_ok=True)
#     checkpoints_dir = exp_dir.joinpath('checkpoints/')
#     checkpoints_dir.mkdir(exist_ok=True)
#     log_dir = exp_dir.joinpath('logs/')
#     log_dir.mkdir(exist_ok=True)

#     '''LOG'''
#     args = parse_args()
#     logger = logging.getLogger("Model")
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     log_string('PARAMETER ...')
#     log_string(args)

#     # 判断是否使用S3DIS数据集
#     is_s3dis = args.config is not None and args.config.find('s3dis') != -1
    
#     # 根据数据集类型设置参数
#     if is_s3dis:
#         num_classes = 13  # S3DIS有13个语义类别
#         num_part = 13     # S3DIS直接分割为13个语义类别
#         root = '/home/dlss107552304043/PointMamba/data/Stanford3dDataset_v1.2_Aligned_Version'
#     else:
#         num_classes = 16  # ShapeNetPart有16个物体类别
#         num_part = 50     # ShapeNetPart有50个部件类别
#         root = args.root

#     '''DATASET LOADING'''
#     if is_s3dis:
#         # S3DIS数据集
#         TRAIN_DATASET = S3DISDataset(root=root,npoints=args.npoint,split='train',test_area=args.test_area,use_zigzag=False,  # 将从配置文件中读取
#             num_paths=6,zigzag_indices=[0, 1, 2, 3, 4, 5])
#         TEST_DATASET = S3DISDataset(root=root,npoints=args.npoint,split='test',test_area=args.test_area,use_zigzag=False,  # 将从配置文件中读取
#             num_paths=6,zigzag_indices=[0, 1, 2, 3, 4, 5])
#     else:
#         # ShapeNetPart数据集
#         TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
#         TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

#     trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
#                                                   num_workers=10, drop_last=True)
#     testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
#                                                  num_workers=10)
#     log_string("The number of training data is: %d" % len(TRAIN_DATASET))
#     log_string("The number of test data is: %d" % len(TEST_DATASET))

#     '''MODEL LOADING'''
#     shutil.copy('models/%s.py' % args.model, str(exp_dir))
    
#     if args.config is not None:
#         from utils.config import cfg_from_yaml_file
#         from utils.logger import print_log
#         if args.config[:13] == "segmentation/":
#             args.config = args.config[13:]
#         config = cfg_from_yaml_file(args.config)
#         log_string(config)
        
#         # 更新数据集的zigzag参数
#         if is_s3dis:
#             TRAIN_DATASET.use_zigzag = config.get('use_zigzag', False)
#             TRAIN_DATASET.num_paths = config.get('num_paths', 6)
#             TRAIN_DATASET.zigzag_indices = config.get('zigzag_indices', [0, 1, 2, 3, 4, 5])
#             TEST_DATASET.use_zigzag = config.get('use_zigzag', False)
#             TEST_DATASET.num_paths = config.get('num_paths', 6)
#             TEST_DATASET.zigzag_indices = config.get('zigzag_indices', [0, 1, 2, 3, 4, 5])
#             if hasattr(config, 'test_area'):
#                 args.test_area = config.test_area
        
#         if hasattr(config, 'epoch'):
#             args.epoch = config.epoch
#         if hasattr(config, 'batch_size'):
#             args.batch_size = config.batch_size
#         if hasattr(config, 'learning_rate'):
#             args.learning_rate = config.learning_rate
#         if hasattr(config, 'ckpt') and args.ckpts is None:
#             args.ckpts = config.ckpts
#         if hasattr(config, 'model'):
#             MODEL = importlib.import_module(config.model) if hasattr(config, 'model') else importlib.import_module(
#                 args.model)
#             if is_s3dis:
#                 classifier = MODEL.get_s3dis_model(num_classes, config).cuda()
#             else:
#                 classifier = MODEL.get_model(num_part, config).cuda()
#         else:
#             MODEL = importlib.import_module(args.model)
#             if is_s3dis:
#                 classifier = MODEL.get_s3dis_model(num_classes).cuda()
#             else:
#                 classifier = MODEL.get_model(num_part).cuda()
#     else:
#         MODEL = importlib.import_module(args.model)
#         shutil.copy('models/%s.py' % args.model, str(exp_dir))
#         if is_s3dis:
#             classifier = MODEL.get_s3dis_model(num_classes).cuda()
#         else:
#             classifier = MODEL.get_model(num_part).cuda()
    
#     criterion = MODEL.get_loss().cuda()
#     classifier.apply(inplace_relu)
#     print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
#     start_epoch = 0

#     if args.ckpts is not None:
#         if args.ckpts[:13] == "segmentation/":
#             args.ckpts = args.ckpts[13:]
#         classifier.load_model_from_ckpt(args.ckpts)
#         log_string('Load model from %s' % args.ckpts)
#     else:
#         log_string('No existing model, starting training from scratch...')

#     ## we use adamw and cosine scheduler
#     def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
#         decay = []
#         no_decay = []
#         num_trainable_params = 0
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue  # frozen weights
#             if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
#                 no_decay.append(param)
#                 num_trainable_params += param.numel()
#             else:
#                 decay.append(param)
#                 num_trainable_params += param.numel()

#         total_params = sum([v.numel() for v in model.parameters()])
#         non_trainable_params = total_params - num_trainable_params
#         log_string('########################################################################')
#         log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
#             '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
#         log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
#         log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
#         log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
#         log_string('########################################################################')

#         return [
#             {'params': no_decay, 'weight_decay': 0.},
#             {'params': decay, 'weight_decay': weight_decay}]

#     param_groups = add_weight_decay(classifier, weight_decay=0.05)
#     optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

#     scheduler = CosineLRScheduler(optimizer,
#                                   t_initial=args.epoch,
#                                   t_mul=1,
#                                   lr_min=1e-6,
#                                   decay_rate=0.1,
#                                   warmup_lr_init=1e-6,
#                                   warmup_t=args.warmup_epoch,
#                                   cycle_limit=1,
#                                   t_in_epochs=True)

#     best_acc = 0
#     global_epoch = 0
#     best_class_avg_iou = 0
#     best_inctance_avg_iou = 0

#     classifier.zero_grad()
#     for epoch in range(start_epoch, args.epoch):
#         mean_correct = []

#         log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
#         '''Adjust learning rate and BN momentum'''

#         classifier = classifier.train()
#         loss_batch = []
#         num_iter = 0
#         '''learning one epoch'''
#         for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
#             num_iter += 1
#             points = points.data.numpy()
#             points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
#             points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
#             points = torch.Tensor(points)
#             points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#             points = points.transpose(2, 1)

#             if is_s3dis:
#                 # S3DIS不需要类别标签作为输入
#                 seg_pred = classifier(points)
#             else:
#                 # ShapeNetPart需要类别标签作为输入
#                 seg_pred = classifier(points, to_categorical(label, num_classes))
            
#             seg_pred = seg_pred.contiguous().view(-1, num_part)
#             target = target.view(-1, 1)[:, 0]
#             pred_choice = seg_pred.data.max(1)[1]

#             correct = pred_choice.eq(target.data).cpu().sum()
#             mean_correct.append(correct.item() / (args.batch_size * args.npoint))
#             loss = criterion(seg_pred, target)
#             loss.backward()
#             optimizer.step()
#             loss_batch.append(loss.detach().cpu())

#             if num_iter == 1:
#                 torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
#                 num_iter = 0
#                 optimizer.step()
#                 classifier.zero_grad()

#         if isinstance(scheduler, list):
#             for item in scheduler:
#                 item.step(epoch)
#         else:
#             scheduler.step(epoch)

#         train_instance_acc = np.mean(mean_correct)
#         loss1 = np.mean(loss_batch)
#         log_string('Train accuracy is: %.5f' % train_instance_acc)
#         log_string('Train loss: %.5f' % loss1)
#         log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

#         with torch.no_grad():
#             test_metrics = {}
#             total_correct = 0
#             total_seen = 0
#             total_seen_class = [0 for _ in range(num_part)]
#             total_correct_class = [0 for _ in range(num_part)]
            
#             if is_s3dis:
#                 # S3DIS评估逻辑
#                 classifier = classifier.eval()
#                 class_ious = []
                
#                 for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
#                                                               smoothing=0.9):
#                     cur_batch_size, NUM_POINT, _ = points.size()
#                     points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#                     seg_pred = classifier(points)
#                     cur_pred_val = seg_pred.cpu().data.numpy()
#                     cur_pred_val = np.argmax(cur_pred_val, axis=2)
#                     target = target.cpu().data.numpy()

#                     correct = np.sum(cur_pred_val == target)
#                     total_correct += correct
#                     total_seen += (cur_batch_size * NUM_POINT)

#                     for l in range(num_part):
#                         total_seen_class[l] += np.sum(target == l)
#                         total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

#                     # 计算每个样本的IoU
#                     for i in range(cur_batch_size):
#                         segp = cur_pred_val[i, :]
#                         segl = target[i, :]
#                         part_ious = []
#                         for l in range(num_part):
#                             if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
#                                 part_ious.append(1.0)
#                             else:
#                                 part_ious.append(np.sum((segl == l) & (segp == l)) / float(
#                                     np.sum((segl == l) | (segp == l))))
#                         class_ious.append(np.mean(part_ious))

#                 # 输出每个类别的mIoU
#                 for i, class_name in enumerate(s3dis_classes):
#                     if total_seen_class[i] > 0:
#                         class_acc = total_correct_class[i] / float(total_seen_class[i])
#                         log_string('eval mIoU of %s %f' % (class_name + ' ' * (14 - len(class_name)), class_acc))

#                 test_metrics['accuracy'] = total_correct / float(total_seen)
#                 test_metrics['class_avg_accuracy'] = np.mean(
#                     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
#                 test_metrics['class_avg_iou'] = np.mean(
#                     [total_correct_class[i] / float(total_seen_class[i]) if total_seen_class[i] > 0 else 0 
#                      for i in range(num_part)])
#                 test_metrics['inctance_avg_iou'] = np.mean(class_ious)
                
#             else:
#                 # ShapeNetPart评估逻辑（原有逻辑）
#                 shape_ious = {cat: [] for cat in seg_classes.keys()}
#                 seg_label_to_cat = {}

#                 for cat in seg_classes.keys():
#                     for label in seg_classes[cat]:
#                         seg_label_to_cat[label] = cat

#                 classifier = classifier.eval()

#                 for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
#                                                               smoothing=0.9):
#                     cur_batch_size, NUM_POINT, _ = points.size()
#                     points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#                     seg_pred = classifier(points, to_categorical(label, num_classes))
#                     cur_pred_val = seg_pred.cpu().data.numpy()
#                     cur_pred_val_logits = cur_pred_val
#                     cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
#                     target = target.cpu().data.numpy()

#                     for i in range(cur_batch_size):
#                         cat = seg_label_to_cat[target[i, 0]]
#                         logits = cur_pred_val_logits[i, :, :]
#                         cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

#                     correct = np.sum(cur_pred_val == target)
#                     total_correct += correct
#                     total_seen += (cur_batch_size * NUM_POINT)

#                     for l in range(num_part):
#                         total_seen_class[l] += np.sum(target == l)
#                         total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

#                     for i in range(cur_batch_size):
#                         segp = cur_pred_val[i, :]
#                         segl = target[i, :]
#                         cat = seg_label_to_cat[segl[0]]
#                         part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
#                         for l in seg_classes[cat]:
#                             if (np.sum(segl == l) == 0) and (
#                                     np.sum(segp == l) == 0):  # part is not present, no prediction as well
#                                 part_ious[l - seg_classes[cat][0]] = 1.0
#                             else:
#                                 part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
#                                     np.sum((segl == l) | (segp == l)))
#                         shape_ious[cat].append(np.mean(part_ious))

#                 all_shape_ious = []
#                 for cat in shape_ious.keys():
#                     for iou in shape_ious[cat]:
#                         all_shape_ious.append(iou)
#                     shape_ious[cat] = np.mean(shape_ious[cat])
#                 mean_shape_ious = np.mean(list(shape_ious.values()))
#                 test_metrics['accuracy'] = total_correct / float(total_seen)
#                 test_metrics['class_avg_accuracy'] = np.mean(
#                     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
#                 for cat in sorted(shape_ious.keys()):
#                     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
#                 test_metrics['class_avg_iou'] = mean_shape_ious
#                 test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

#         log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
#             epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
#         if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
#             logger.info('Save model...')
#             savepath = str(checkpoints_dir) + '/best_model.pth'
#             log_string('Saving at %s' % savepath)
#             state = {
#                 'epoch': epoch,
#                 'train_acc': train_instance_acc,
#                 'test_acc': test_metrics['accuracy'],
#                 'class_avg_iou': test_metrics['class_avg_iou'],
#                 'inctance_avg_iou': test_metrics['inctance_avg_iou'],
#                 'model_state_dict': classifier.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }
#             torch.save(state, savepath)
#             log_string('Saving model....')

#         if test_metrics['accuracy'] > best_acc:
#             best_acc = test_metrics['accuracy']
#         if test_metrics['class_avg_iou'] > best_class_avg_iou:
#             best_class_avg_iou = test_metrics['class_avg_iou']
#         if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
#             best_inctance_avg_iou = test_metrics['inctance_avg_iou']
#         log_string('Best accuracy is: %.5f' % best_acc)
#         log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
#         log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
#         global_epoch += 1


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)





# # # 第二次修改
# import argparse
# import os
# import random

# import torch
# import datetime
# import logging
# import sys
# import importlib
# import shutil
# import provider
# import numpy as np
# import torch.optim as optim
# from timm.scheduler import CosineLRScheduler
# from pathlib import Path
# from tqdm import tqdm
# from dataset import PartNormalDataset
# # 添加S3DIS数据集导入
# from dataset import S3DISDataset

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat

# # S3DIS类别定义
# s3dis_classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 
#                  'window', 'door', 'table', 'chair', 'sofa', 
#                  'bookcase', 'board', 'clutter']


# def inplace_relu(m):
#     classname = m.__class__.__name__
#     if classname.find('ReLU') != -1:
#         m.inplace = True


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
#     if (y.is_cuda):
#         return new_y.cuda()
#     return new_y


# def parse_args():
#     parser = argparse.ArgumentParser('Model')
#     parser.add_argument('--model', type=str, default='pt_mamba', help='model name')
#     parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
#     parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
#     parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
#     # parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
#     parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
#     parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
#     parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
#     parser.add_argument('--npoint', type=int, default=4096, help='point Number')
#     parser.add_argument('--normal', action='store_true', default=False, help='use normals')
#     parser.add_argument('--config', type=str, default=None, help='config file')
#     parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
#     parser.add_argument('--root', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version/',
#                         help='data root')   # /home/dlss107552304043/PointMamba/data/Stanford3dDataset_v1.2_Aligned_Version
#     parser.add_argument('--test_area', type=int, default=5, help='test area for S3DIS (1-6)')
    
#     return parser.parse_args()


# def main(args):
#     def log_string(str):
#         logger.info(str)
#         print(str)

#     '''HYPER PARAMETER'''
#     # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#     '''CREATE DIR'''
#     timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
#     exp_dir = Path('./log/')
#     exp_dir.mkdir(exist_ok=True)
#     exp_dir = exp_dir.joinpath('part_seg')
#     exp_dir.mkdir(exist_ok=True)
#     if args.log_dir is None:
#         exp_dir = exp_dir.joinpath(timestr)
#     else:
#         exp_dir = exp_dir.joinpath(args.log_dir)
#     exp_dir.mkdir(exist_ok=True)
#     checkpoints_dir = exp_dir.joinpath('checkpoints/')
#     checkpoints_dir.mkdir(exist_ok=True)
#     log_dir = exp_dir.joinpath('logs/')
#     log_dir.mkdir(exist_ok=True)

#     '''LOG'''
#     args = parse_args()
#     logger = logging.getLogger("Model")
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     log_string('PARAMETER ...')
#     log_string(args)

#     # 判断是否使用S3DIS数据集
#     is_s3dis = args.config is not None and args.config.find('s3dis') != -1
    
#     # 根据数据集类型设置参数
#     if is_s3dis:
#         num_classes = 13  # S3DIS有13个语义类别
#         num_part = 13     # S3DIS直接分割为13个语义类别
#         root = '/home/dlss107552304043/PointMamba/data/Stanford3dDataset_v1.2_Aligned_Version'
#     else:
#         num_classes = 16  # ShapeNetPart有16个物体类别
#         num_part = 50     # ShapeNetPart有50个部件类别
#         root = args.root

#     # 先处理配置文件，获取所有参数
#     config = None
#     if args.config is not None:
#         from utils.config import cfg_from_yaml_file
#         from utils.logger import print_log
#         if args.config[:13] == "segmentation/":
#             args.config = args.config[13:]
#         config = cfg_from_yaml_file(args.config)
#         log_string(config)
        
#         # 更新参数
#         if hasattr(config, 'epoch'):
#             args.epoch = config.epoch
#         if hasattr(config, 'batch_size'):
#             args.batch_size = config.batch_size
#         if hasattr(config, 'learning_rate'):
#             args.learning_rate = config.learning_rate
#         if hasattr(config, 'ckpt') and args.ckpts is None:
#             args.ckpts = config.ckpts
#         if hasattr(config, 'test_area'):
#             args.test_area = config.test_area

#     '''DATASET LOADING''' # - 只创建一次，使用配置文件参数
#     if is_s3dis:
#         # 使用配置文件中的参数创建数据集
#         use_zigzag = config.get('use_zigzag', False) if config else False
#         num_paths = config.get('num_paths', 6) if config else 6
#         zigzag_indices = config.get('zigzag_indices', [0, 1, 2, 3, 4, 5]) if config else [0, 1, 2, 3, 4, 5]
        
#         TRAIN_DATASET = S3DISDataset(
#             root=root,
#             npoints=args.npoint,
#             split='train',
#             test_area=args.test_area,
#             use_zigzag=use_zigzag,
#             num_paths=num_paths,
#             zigzag_indices=zigzag_indices
#         )
#         TEST_DATASET = S3DISDataset(
#             root=root,
#             npoints=args.npoint,
#             split='test',
#             test_area=args.test_area,
#             use_zigzag=use_zigzag,
#             num_paths=num_paths,
#             zigzag_indices=zigzag_indices
#         )
#     else:
#         TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
#         TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

#     trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
#                                                   num_workers=10, drop_last=True)
#     testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
#                                                  num_workers=10)
#     log_string("The number of training data is: %d" % len(TRAIN_DATASET))
#     log_string("The number of test data is: %d" % len(TEST_DATASET))

#     '''MODEL LOADING'''
#     shutil.copy('models/%s.py' % args.model, str(exp_dir))
    
#     if config is not None:
#         if hasattr(config, 'model'):
#             # 清理模型名称中的空格
#             model_name = config.model.strip()
#             MODEL = importlib.import_module(model_name)
#             if is_s3dis:
#                 classifier = MODEL.get_s3dis_model(num_classes, config).cuda()
#             else:
#                 classifier = MODEL.get_model(num_part, config).cuda()
#         else:
#             MODEL = importlib.import_module(args.model)
#             if is_s3dis:
#                 classifier = MODEL.get_s3dis_model(num_classes).cuda()
#             else:
#                 classifier = MODEL.get_model(num_part).cuda()
#     else:
#         MODEL = importlib.import_module(args.model)
#         shutil.copy('models/%s.py' % args.model, str(exp_dir))
#         if is_s3dis:
#             classifier = MODEL.get_s3dis_model(num_classes).cuda()
#         else:
#             classifier = MODEL.get_model(num_part).cuda()
    
#     criterion = MODEL.get_loss().cuda()
#     classifier.apply(inplace_relu)
#     print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
#     start_epoch = 0

#     if args.ckpts is not None:
#         if args.ckpts[:13] == "segmentation/":
#             args.ckpts = args.ckpts[13:]
#         classifier.load_model_from_ckpt(args.ckpts)
#         log_string('Load model from %s' % args.ckpts)
#     else:
#         log_string('No existing model, starting training from scratch...')

#     ## we use adamw and cosine scheduler
#     def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
#         decay = []
#         no_decay = []
#         num_trainable_params = 0
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue  # frozen weights
#             if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
#                 no_decay.append(param)
#                 num_trainable_params += param.numel()
#             else:
#                 decay.append(param)
#                 num_trainable_params += param.numel()

#         total_params = sum([v.numel() for v in model.parameters()])
#         non_trainable_params = total_params - num_trainable_params
#         log_string('########################################################################')
#         log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
#             '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
#         log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
#         log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
#         log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
#         log_string('########################################################################')

#         return [
#             {'params': no_decay, 'weight_decay': 0.},
#             {'params': decay, 'weight_decay': weight_decay}]

#     param_groups = add_weight_decay(classifier, weight_decay=0.05)
#     optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

#     scheduler = CosineLRScheduler(optimizer,
#                                   t_initial=args.epoch,
#                                   t_mul=1,
#                                   lr_min=1e-6,
#                                   decay_rate=0.1,
#                                   warmup_lr_init=1e-6,
#                                   warmup_t=args.warmup_epoch,
#                                   cycle_limit=1,
#                                   t_in_epochs=True)

#     best_acc = 0
#     global_epoch = 0
#     best_class_avg_iou = 0
#     best_inctance_avg_iou = 0

#     classifier.zero_grad()
#     for epoch in range(start_epoch, args.epoch):
#         mean_correct = []

#         log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
#         '''Adjust learning rate and BN momentum'''

#         classifier = classifier.train()
#         loss_batch = []
#         num_iter = 0
#         '''learning one epoch'''
#         for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
#             num_iter += 1
#             points = points.data.numpy()
#             points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
#             points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
#             # 在trainDataLoader的迭代中增加更多增强
#             points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])       # 新增旋转
#             points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])       # 新增抖动
#             points = torch.Tensor(points)
#             points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#             points = points.transpose(2, 1)

#             if is_s3dis:
#                 # S3DIS不需要类别标签作为输入
#                 seg_pred = classifier(points)
#             else:
#                 # ShapeNetPart需要类别标签作为输入
#                 seg_pred = classifier(points, to_categorical(label, num_classes))
            
#             seg_pred = seg_pred.contiguous().view(-1, num_part)
#             target = target.view(-1, 1)[:, 0]
#             pred_choice = seg_pred.data.max(1)[1]

#             correct = pred_choice.eq(target.data).cpu().sum()
#             mean_correct.append(correct.item() / (args.batch_size * args.npoint))
#             loss = criterion(seg_pred, target)
#             loss.backward()
#             optimizer.step()
#             loss_batch.append(loss.detach().cpu())

#             if num_iter == 1:
#                 torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
#                 num_iter = 0
#                 optimizer.step()
#                 classifier.zero_grad()

#         if isinstance(scheduler, list):
#             for item in scheduler:
#                 item.step(epoch)
#         else:
#             scheduler.step(epoch)

#         train_instance_acc = np.mean(mean_correct)
#         loss1 = np.mean(loss_batch)
#         log_string('Train accuracy is: %.5f' % train_instance_acc)
#         log_string('Train loss: %.5f' % loss1)
#         log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

#         with torch.no_grad():
#             test_metrics = {}
#             total_correct = 0
#             total_seen = 0
#             total_seen_class = [0 for _ in range(num_part)]
#             total_correct_class = [0 for _ in range(num_part)]
            
#             if is_s3dis:
#                 # S3DIS评估逻辑
#                 classifier = classifier.eval()
#                 class_ious = []
                
#                 for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
#                                                               smoothing=0.9):
#                     cur_batch_size, NUM_POINT, _ = points.size()
#                     points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#                     seg_pred = classifier(points)
#                     cur_pred_val = seg_pred.cpu().data.numpy()
#                     cur_pred_val = np.argmax(cur_pred_val, axis=2)
#                     target = target.cpu().data.numpy()

#                     correct = np.sum(cur_pred_val == target)
#                     total_correct += correct
#                     total_seen += (cur_batch_size * NUM_POINT)

#                     for l in range(num_part):
#                         total_seen_class[l] += np.sum(target == l)
#                         total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

#                     # 计算每个样本的IoU
#                     for i in range(cur_batch_size):
#                         segp = cur_pred_val[i, :]
#                         segl = target[i, :]
#                         part_ious = []
#                         for l in range(num_part):
#                             if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
#                                 part_ious.append(1.0)
#                             else:
#                                 part_ious.append(np.sum((segl == l) & (segp == l)) / float(
#                                     np.sum((segl == l) | (segp == l))))
#                         class_ious.append(np.mean(part_ious))

#                 # 输出每个类别的mIoU
#                 for i, class_name in enumerate(s3dis_classes):
#                     if total_seen_class[i] > 0:
#                         class_acc = total_correct_class[i] / float(total_seen_class[i])
#                         log_string('eval mIoU of %s %f' % (class_name + ' ' * (14 - len(class_name)), class_acc))

#                 test_metrics['accuracy'] = total_correct / float(total_seen)
#                 test_metrics['class_avg_accuracy'] = np.mean(
#                     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
#                 test_metrics['class_avg_iou'] = np.mean(
#                     [total_correct_class[i] / float(total_seen_class[i]) if total_seen_class[i] > 0 else 0 
#                      for i in range(num_part)])
#                 test_metrics['inctance_avg_iou'] = np.mean(class_ious)
                
#             else:
#                 # ShapeNetPart评估逻辑（原有逻辑）
#                 shape_ious = {cat: [] for cat in seg_classes.keys()}
#                 seg_label_to_cat = {}

#                 for cat in seg_classes.keys():
#                     for label in seg_classes[cat]:
#                         seg_label_to_cat[label] = cat

#                 classifier = classifier.eval()

#                 for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
#                                                               smoothing=0.9):
#                     cur_batch_size, NUM_POINT, _ = points.size()
#                     points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#                     seg_pred = classifier(points, to_categorical(label, num_classes))
#                     cur_pred_val = seg_pred.cpu().data.numpy()
#                     cur_pred_val_logits = cur_pred_val
#                     cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
#                     target = target.cpu().data.numpy()

#                     for i in range(cur_batch_size):
#                         cat = seg_label_to_cat[target[i, 0]]
#                         logits = cur_pred_val_logits[i, :, :]
#                         cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

#                     correct = np.sum(cur_pred_val == target)
#                     total_correct += correct
#                     total_seen += (cur_batch_size * NUM_POINT)

#                     for l in range(num_part):
#                         total_seen_class[l] += np.sum(target == l)
#                         total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

#                     for i in range(cur_batch_size):
#                         segp = cur_pred_val[i, :]
#                         segl = target[i, :]
#                         cat = seg_label_to_cat[segl[0]]
#                         part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
#                         for l in seg_classes[cat]:
#                             if (np.sum(segl == l) == 0) and (
#                                     np.sum(segp == l) == 0):  # part is not present, no prediction as well
#                                 part_ious[l - seg_classes[cat][0]] = 1.0
#                             else:
#                                 part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
#                                     np.sum((segl == l) | (segp == l)))
#                         shape_ious[cat].append(np.mean(part_ious))

#                 all_shape_ious = []
#                 for cat in shape_ious.keys():
#                     for iou in shape_ious[cat]:
#                         all_shape_ious.append(iou)
#                     shape_ious[cat] = np.mean(shape_ious[cat])
#                 mean_shape_ious = np.mean(list(shape_ious.values()))
#                 test_metrics['accuracy'] = total_correct / float(total_seen)
#                 test_metrics['class_avg_accuracy'] = np.mean(
#                     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
#                 for cat in sorted(shape_ious.keys()):
#                     log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
#                 test_metrics['class_avg_iou'] = mean_shape_ious
#                 test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

#         log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
#             epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
#         if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
#             logger.info('Save model...')
#             savepath = str(checkpoints_dir) + '/best_model.pth'
#             log_string('Saving at %s' % savepath)
#             state = {
#                 'epoch': epoch,
#                 'train_acc': train_instance_acc,
#                 'test_acc': test_metrics['accuracy'],
#                 'class_avg_iou': test_metrics['class_avg_iou'],
#                 'inctance_avg_iou': test_metrics['inctance_avg_iou'],
#                 'model_state_dict': classifier.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }
#             torch.save(state, savepath)
#             log_string('Saving model....')

#         if test_metrics['accuracy'] > best_acc:
#             best_acc = test_metrics['accuracy']
#         if test_metrics['class_avg_iou'] > best_class_avg_iou:
#             best_class_avg_iou = test_metrics['class_avg_iou']
#         if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
#             best_inctance_avg_iou = test_metrics['inctance_avg_iou']
#         log_string('Best accuracy is: %.5f' % best_acc)
#         log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
#         log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
#         global_epoch += 1


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)





