from typing import Union, Optional
import torch.nn.functional as F
import math
import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from knn_cuda import KNN
from .block import Block
from .build import MODELS

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
   
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)

        # reordering strategy
        center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
        center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
        group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
            1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
            1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
            1, 1, group_input_tokens.shape[-1])))
        pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
        pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
        pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
        group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],
                                       dim=1)
        pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        x = group_input_tokens
        # transformer
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = x[:, :].mean(1)
        ret = self.cls_head_finetune(concat_f)
        return ret



    # ✅ 新增：特征提取方法
    def get_features(self, pts):
        """
        提取全局特征（不经过分类头）
        
        Args:
            pts: 输入点云 [B, N, 3]
        
        Returns:
            features: 全局特征 [B, trans_dim]
        """
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)
        
        # reordering strategy
        center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
        center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
        
        group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (1, 1, group_input_tokens.shape[-1])))
        group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (1, 1, group_input_tokens.shape[-1])))
        
        pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
        pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
        pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
        
        group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z], dim=1)
        pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
        
        x = group_input_tokens
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)
        
        # 返回全局特征（不经过分类头）
        features = x[:, :].mean(1)  # [B, trans_dim]
        
        # L2 归一化（让聚类更紧凑）
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features

# # 并行掩码——改进1+2
class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        # 传统掩码率-用于随机掩码和块掩码
        self.traditional_mask_ratio = config.transformer_config.mask_ratio
        
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        
        # 语义分组参数
        self.semantic_groups = config.transformer_config.get('semantic_groups', 4)  # K=4
        self.use_semantic_grouping = config.transformer_config.get('use_semantic_grouping', False)
        
        # 并行掩码率 - 只和K有关，计算公式为 (K-1)/K
        if self.use_semantic_grouping:
            self.parallel_mask_ratio = (self.semantic_groups - 1) / self.semantic_groups
            print_log(f'[MaskMamba] Parallel mask ratio (K-1)/K = ({self.semantic_groups}-1)/{self.semantic_groups} = {self.parallel_mask_ratio:.3f}', 
                     logger='Mamba')
        else:
            self.parallel_mask_ratio = None
        
        print_log(f'[MaskMamba] Traditional mask ratio: {self.traditional_mask_ratio}', logger='Mamba')
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        
        # 用于保存相似度信息
        self.group_similarities = None
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.traditional_mask_ratio == 0:  # 使用traditional_mask_ratio
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                        dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.traditional_mask_ratio  # 使用traditional_mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.traditional_mask_ratio == 0:  # 使用traditional_mask_ratio
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.traditional_mask_ratio * G)  # 使用traditional_mask_ratio

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G
    
    # 动态使用K值的变化
    def _semantic_grouping(self, group_input_tokens, center):
        """
        动态语义分组，支持任意K值
        
        Args:
            group_input_tokens: [B, G, C] 
            center: [B, G, 3]
        
        Returns:
            semantic_tokens: [B, K, G//K, C]
            semantic_centers: [B, K, G//K, 3]
            group_indices: [B, K, G//K] - 每个token的原始索引
            group_similarities: [B, K] - 每组与anchor的平均相似度
        """
        B, G, C = group_input_tokens.shape
        K = self.semantic_groups
        group_size = G // K  # 每组的token数量
        
        # 初始化输出
        semantic_tokens = torch.zeros(B, K, group_size, C, device=group_input_tokens.device)
        semantic_centers = torch.zeros(B, K, group_size, 3, device=center.device)
        group_indices = torch.zeros(B, K, group_size, dtype=torch.long, device=group_input_tokens.device)
        group_similarities = torch.zeros(B, K, device=group_input_tokens.device)
        
        for b in range(B):
            # Step 1: 随机选择 group_size 个token作为anchor group
            anchor_indices = torch.randperm(G, device=group_input_tokens.device)[:group_size]
            anchor_tokens = group_input_tokens[b, anchor_indices]  # [group_size, C]
            anchor_centers = center[b, anchor_indices]  # [group_size, 3]
            
            # 存储anchor group (第0组)
            semantic_tokens[b, 0] = anchor_tokens
            semantic_centers[b, 0] = anchor_centers
            group_indices[b, 0] = anchor_indices
            group_similarities[b, 0] = 1.0  # anchor自相似度为1
            
            # Step 2: 计算剩余token与anchor的相似度
            remaining_mask = torch.ones(G, dtype=torch.bool, device=group_input_tokens.device)
            remaining_mask[anchor_indices] = False
            remaining_indices = torch.arange(G, device=group_input_tokens.device)[remaining_mask]
            remaining_tokens = group_input_tokens[b, remaining_indices]  # [G-group_size, C]
            remaining_centers = center[b, remaining_indices]
            
            # 归一化特征
            anchor_norm = F.normalize(anchor_tokens, p=2, dim=-1)
            remaining_norm = F.normalize(remaining_tokens, p=2, dim=-1)
            
            # 计算每个剩余token与所有anchor的最大相似度
            similarity_matrix = torch.matmul(remaining_norm, anchor_norm.T)  # [G-group_size, group_size]
            max_similarity, _ = similarity_matrix.max(dim=1)  # [G-group_size]
            
            # Step 3: 根据相似度排序并分配到K-1个组
            sorted_indices = torch.argsort(max_similarity, descending=True)
            
            for k in range(1, K):
                start_idx = (k - 1) * group_size
                end_idx = k * group_size
                
                # 选择这一组的token
                group_sorted_indices = sorted_indices[start_idx:end_idx]
                group_token_indices = remaining_indices[group_sorted_indices]
                
                semantic_tokens[b, k] = remaining_tokens[group_sorted_indices]
                semantic_centers[b, k] = remaining_centers[group_sorted_indices]
                group_indices[b, k] = group_token_indices
                
                # 计算这一组的平均相似度
                group_similarities[b, k] = max_similarity[group_sorted_indices].mean()
        
        return semantic_tokens, semantic_centers, group_indices, group_similarities




    def _semantic_grouping(self, group_input_tokens, center):
        """
        正确的逻辑：
        1. 随机选择id1 (16个)
        2. 计算16×48相似度矩阵  
        3. id2选择每个位置相似度最高的 (16个)
        4. id3选择每个位置相似度次高的 (16个)
        5. id4就是剩余的16个
        """
        B, G, C = group_input_tokens.shape
        K = self.semantic_groups  # 4
        tokens_per_group = G // K  # 16
        
        semantic_tokens = torch.zeros(B, K, tokens_per_group, C, device=group_input_tokens.device)
        semantic_centers = torch.zeros(B, K, tokens_per_group, 3, device=group_input_tokens.device)
        group_indices = torch.zeros(B, K, tokens_per_group, dtype=torch.long, device=group_input_tokens.device)
        group_similarities = torch.zeros(B, K, device=group_input_tokens.device)
        
        for b in range(B):
            tokens = group_input_tokens[b]  # [64, C]
            centers = center[b]  # [64, 3]
            
            # 1. 随机选择第一组 (id1) - 16个tokens
            all_indices = torch.arange(G, device=tokens.device)
            first_group_idx = all_indices[torch.randperm(G)[:tokens_per_group]]
            first_group_tokens = tokens[first_group_idx]  # [16, C]
            first_group_centers = centers[first_group_idx]  # [16, 3]
            
            semantic_tokens[b, 0] = first_group_tokens
            semantic_centers[b, 0] = first_group_centers
            group_indices[b, 0] = first_group_idx
            group_similarities[b, 0] = 1.0
            
            # 2. 剩余48个tokens
            remaining_mask = torch.ones(G, dtype=torch.bool, device=tokens.device)
            remaining_mask[first_group_idx] = False
            remaining_idx = all_indices[remaining_mask]  # [48]
            remaining_tokens = tokens[remaining_idx]  # [48, C]
            remaining_centers = centers[remaining_idx]  # [48, 3]
            
            # 3. 计算16×48的语义相似度矩阵
            first_norm = F.normalize(first_group_tokens, p=2, dim=1)  # [16, C]
            remaining_norm = F.normalize(remaining_tokens, p=2, dim=1)  # [48, C]
            similarity_matrix = torch.mm(first_norm, remaining_norm.t())  # [16, 48]
            
            # 4. 为每个剩余token计算与第一组的最大相似度，并排序
            max_similarities = similarity_matrix.max(dim=0)[0]  # [48] - 每个剩余token的最大相似度
            sorted_indices = max_similarities.argsort(descending=True)  # 按相似度降序排列
            
            # 5. id2: 前16个最相似的tokens
            id2_indices = sorted_indices[:tokens_per_group]  # [16]
            id2_similarities = max_similarities[id2_indices]
            
            semantic_tokens[b, 1] = remaining_tokens[id2_indices]
            semantic_centers[b, 1] = remaining_centers[id2_indices]
            group_indices[b, 1] = remaining_idx[id2_indices]
            group_similarities[b, 1] = id2_similarities.mean()
            
            # 6. id3: 中间16个次相似的tokens
            id3_indices = sorted_indices[tokens_per_group:2*tokens_per_group]  # [16]
            id3_similarities = max_similarities[id3_indices]
            
            semantic_tokens[b, 2] = remaining_tokens[id3_indices]
            semantic_centers[b, 2] = remaining_centers[id3_indices]
            group_indices[b, 2] = remaining_idx[id3_indices]
            group_similarities[b, 2] = id3_similarities.mean()
            
            # 7. id4: 最后16个最不相似的tokens
            id4_indices = sorted_indices[2*tokens_per_group:]  # [16] - 正好剩余16个
            id4_similarities = max_similarities[id4_indices]
            
            semantic_tokens[b, 3] = remaining_tokens[id4_indices]
            semantic_centers[b, 3] = remaining_centers[id4_indices]
            group_indices[b, 3] = remaining_idx[id4_indices]
            group_similarities[b, 3] = id4_similarities.mean()
        
        return semantic_tokens, semantic_centers, group_indices, group_similarities

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        elif self.mask_type == 'block':
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)
        elif self.mask_type == 'semantic' and self.use_semantic_grouping:
            # 使用改进的分组并行掩码
            group_input_tokens = self.encoder(neighborhood)  # B G C
            semantic_tokens, semantic_centers, group_indices, group_similarities = self._semantic_grouping(group_input_tokens, center)
            
            # 保存相似度信息供Point_MAE_Mamba使用（第二个改进）
            self.group_similarities = group_similarities
            
            B, K, tokens_per_group, C = semantic_tokens.shape
            x_vis_list = []
            bool_masked_pos_list = []
            
            for k in range(K):
                # 每个k的可见tokens: semantic_tokens[:, k]
                x_vis_k = self.blocks(semantic_tokens[:, k].reshape(B, tokens_per_group, C), self.pos_embed(semantic_centers[:, k]))
                x_vis_k = self.norm(x_vis_k)
                
                # mask: 除了这个组，其他都掩码
                bool_masked_pos_k = torch.ones((B, self.config.num_group), dtype=torch.bool, device=center.device)
                bool_masked_pos_k.scatter_(1, group_indices[:, k], False)  # False for visible
                
                x_vis_list.append(x_vis_k)
                bool_masked_pos_list.append(bool_masked_pos_k)
            
            return x_vis_list, bool_masked_pos_list
        else:
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # 默认rand
        
        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)  # B M C
        # add pos embed
        pos = self.pos_embed(center)

        # a random selection of tokens
        pos_vis = pos[~bool_masked_pos].reshape(batch_size, -1, self.trans_dim)
        # finetune x_vis = x_vis + pos_vis

        x_vis = self.blocks(x_vis, pos_vis)  
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# # 并行掩码——改进1+2
@MODELS.register_module()
class Point_MAE_Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            config=config,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        # Add semantic_groups from config
        self.semantic_groups = config.transformer_config.get('semantic_groups', 4)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def _compute_weighted_consistency_loss(self, rec_points_list, group_similarities, K):
        """
        第二个改进：计算加权一致性损失
        Args:
            rec_points_list: K个重建点云列表
            group_similarities: [B, K] 各组与id1的相似度
            K: 组数
        Returns:
            加权一致性损失
        """
        B = group_similarities.shape[0]
        
        loss_consistency = 0
        total_weight = 0
        
        # 计算各组间的权重矩阵
        weights = torch.zeros(K, K, device=group_similarities.device)
        
        for i in range(K):
            for j in range(i+1, K):
                if i == 0:  # id1与其他组
                    # 使用相似度的倒数作为权重，相似度越低权重越高
                    avg_similarity = group_similarities[:, j].mean()
                    weight = 1.0 / (avg_similarity + 1e-6)  # 加小量避免除零
                elif j == 0:  # 其他组与id1（这个分支实际上不会执行，因为i<j）
                    avg_similarity = group_similarities[:, i].mean()
                    weight = 1.0 / (avg_similarity + 1e-6)
                else:  # 其他组间，使用两组与id1相似度的调和平均
                    sim_i = group_similarities[:, i].mean()
                    sim_j = group_similarities[:, j].mean()
                    # 调和平均：更强调较小的相似度值
                    harmonic_mean = 2 * sim_i * sim_j / (sim_i + sim_j + 1e-6)
                    weight = 1.0 / (harmonic_mean + 1e-6)
                
                weights[i, j] = weight
                
                # 计算加权MSE损失
                consistency_loss_ij = F.mse_loss(rec_points_list[i], rec_points_list[j])
                loss_consistency += weight * consistency_loss_ij
                total_weight += weight
        
        # 归一化处理，提高鲁棒性
        if total_weight > 0:
            loss_consistency = loss_consistency / total_weight
        
        # 打印权重信息（用于调试，实际使用时可以注释掉）
        if hasattr(self, '_debug_step'):
            self._debug_step += 1
        else:
            self._debug_step = 0
            
        # if self._debug_step % 100 == 0:  # 每100步打印一次
        #     print_log(f'[DEBUG] Consistency weights - id1-id2: {weights[0,1]:.3f}, id1-id3: {weights[0,2]:.3f}, id1-id4: {weights[0,3]:.3f}', logger='Point_MAE')
        #     print_log(f'[DEBUG] Group similarities - id2: {group_similarities[:, 1].mean():.3f}, id3: {group_similarities[:, 2].mean():.3f}, id4: {group_similarities[:, 3].mean():.3f}', logger='Point_MAE')
        
        return loss_consistency

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis_list_or_x_vis, mask_list_or_mask = self.MAE_encoder(neighborhood, center)

        if self.config.transformer_config.use_semantic_grouping and self.config.transformer_config.mask_type == 'semantic':
            # 处理并行掩码 (K分支)
            x_vis_list = x_vis_list_or_x_vis
            mask_list = mask_list_or_mask
            K = self.semantic_groups
            B = center.shape[0]

            loss_recon = 0
            rec_points_list = []  # 用于一致性损失的 masked 重建

            for k in range(K):
                x_vis = x_vis_list[k]
                bool_masked_pos = mask_list[k]

                pos_emd_vis = self.decoder_pos_embed(center[~bool_masked_pos].reshape(B, -1, 3))
                pos_emd_mask = self.decoder_pos_embed(center[bool_masked_pos].reshape(B, -1, 3))

                _, N, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(B, N, -1)
                x_full = torch.cat([x_vis, mask_token], dim=1)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                x_rec = self.MAE_decoder(x_full, pos_full, N)

                B_rec, M, C = x_rec.shape
                rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B_rec * M, -1, 3)  # B M 3*group_size -> B*M group_size 3

                gt_points = neighborhood[bool_masked_pos].reshape(B_rec * M, -1, 3)
                loss_k = self.loss_func(rebuild_points, gt_points)
                loss_recon += loss_k / K

                # 保存 masked 重建用于一致性 (reshape to B x (masked points) x 3)
                masked_num = self.num_group - (self.num_group // K)  # masked per part
                rec_points_list.append(rebuild_points.reshape(B, masked_num * self.group_size, 3))

            # 第二个改进：加权一致性损失计算
            if hasattr(self.MAE_encoder, 'group_similarities') and self.MAE_encoder.group_similarities is not None:
                group_similarities = self.MAE_encoder.group_similarities
                loss_consistency = self._compute_weighted_consistency_loss(rec_points_list, group_similarities, K)
                # print_log(f'[LOSS] Using weighted consistency loss: {loss_consistency:.6f}', logger='Point_MAE')
            else:
                # 降级到原始一致性损失
                loss_consistency = 0
                count = 0
                for i in range(K):
                    for j in range(i+1, K):
                        loss_consistency += F.mse_loss(rec_points_list[i], rec_points_list[j])
                        count += 1
                if count > 0:
                    loss_consistency /= count
                # print_log(f'[LOSS] Using original consistency loss: {loss_consistency:.6f}', logger='Point_MAE')

            # 总损失
            consistency_weight = 1.0  # 可以通过配置文件调整
            total_loss = loss_recon + consistency_weight * loss_consistency
            
            # print_log(f'[LOSS] Reconstruction loss: {loss_recon:.6f}, Consistency loss: {loss_consistency:.6f}, Total loss: {total_loss:.6f}', logger='Point_MAE')

            if vis:  # visualization (使用第一个分支作为示例)
                mask = mask_list[0]
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - (self.num_group // K)), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)  # 使用最后一个rebuild_points作为示例
                full = torch.cat([full_vis, full_rebuild], dim=0)
                full_center = torch.cat([center[~mask], center[mask]], dim=0)  # 调整顺序
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                return ret1, ret2, full_center
            else:
                return total_loss

        else:
            # 非并行掩码 (原始单分支逻辑)
            x_vis = x_vis_list_or_x_vis
            bool_masked_pos = mask_list_or_mask

            B = x_vis.shape[0]
            pos_emd_vis = self.decoder_pos_embed(center[~bool_masked_pos].reshape(B, -1, 3))
            pos_emd_mask = self.decoder_pos_embed(center[bool_masked_pos].reshape(B, -1, 3))

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = neighborhood[bool_masked_pos].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)

            if vis:  # visualization
                vis_points = neighborhood[~bool_masked_pos].reshape(B * (self.num_group - self.MAE_encoder.num_mask), -1, 3)
                full_vis = vis_points + center[~bool_masked_pos].unsqueeze(1)
                full_rebuild = rebuild_points + center[bool_masked_pos].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                full_center = torch.cat([center[~bool_masked_pos], center[bool_masked_pos]], dim=0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                return ret1, ret2, full_center
            else:
                return loss1

