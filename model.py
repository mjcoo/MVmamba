# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from typing import Optional
from math import log
import numpy as np
from torch.nn import Parameter
from my_mamba import BiDirectionMixerModel, MambaConfig
from KAN import KANLinear
from wavelet import Wavelet



class DynamicBalancedFocalLoss(nn.Module):
    """动态平衡的Focal Loss损失函数
    
    通过动态调整正负样本权重，结合标签平滑和L1正则化，实现更好的分类性能。
    
    参数:
        gamma (float): focal loss的聚焦参数，用于调节难易样本的权重
        alpha (float): 正负样本的平衡因子
        l1_lambda (float): L1正则化系数
        label_smoothing (float): 标签平滑系数
    """
    def __init__(self, gamma=2.0, alpha=0.5, l1_lambda=1e-5, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.label_smoothing = label_smoothing

    def forward(self, y_true, y_pred, model):
        # 动态平衡因子
        pos_weight = (1 - y_true.mean()).detach()
        neg_weight = y_true.mean().detach()
        
        # 标签平滑
        y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Focal Loss计算
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss
        
        # 动态平衡权重
        balance_weight = y_true * pos_weight + (1 - y_true) * neg_weight
        focal_loss = balance_weight * focal_loss
        
        # 正则化
        l1_reg = torch.tensor(0., device=y_pred.device)
        for param in model.parameters():
            l1_reg += torch.norm(param, p=1)
        
        return focal_loss.mean() + self.l1_lambda * l1_reg
class FeatureFusion(nn.Module):
    """可配置的特征融合模块
    
    使用动态门控机制融合多个模态的特征，支持灵活的特征选择。
    
    主要功能:
        - 特征投影：将不同维度的特征映射到统一空间
        - 动态门控：自适应调整不同特征的重要性
        - 特征增强：进一步提升融合特征的表达能力
    """
    def __init__(self, input_dims):
        super().__init__()
        config = Config.MODEL_CONFIG['fusion_config']
        self.active_features = config['active_features']
        
        # 特征投影层
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, config['common_dim'])
            for name, dim in zip(self.active_features, input_dims)
            if name in self.active_features
        })
        
        # 动态门控网络
        n_features = len(self.active_features)
        self.gate = nn.Sequential(
            nn.Linear(config['common_dim'] * n_features, n_features),
            nn.Softmax(dim=-1)
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Linear(config['common_dim'], config['common_dim']*2),
            nn.GLU(),
            nn.LayerNorm(config['common_dim'])
            #nn.Linear(config['common_dim']*2, config['common_dim'])
        )
    
    def forward(self, features):
        # 处理激活的特征
        active_features = {
            name: feat for name, feat in zip(Config.MODEL_CONFIG['input_dims'].keys(), features)
            if name in self.active_features
        }       
        # 特征投影
        proj_features = {
            name: self.projections[name](feat)
            for name, feat in active_features.items()
        }
        origin = torch.stack(list(proj_features.values()), dim=1).sum(dim=1)
        # 将所有投影特征拼接起来用于计算门控权重
        concat_features = torch.cat(list(proj_features.values()), dim=-1)
        weights = self.gate(concat_features)  # [batch_size, n_features]
        
        # 堆叠所有特征并应用权重
        stacked_features = torch.stack(list(proj_features.values()), dim=1)  # [batch_size, n_features, hidden_dim]
        weights = weights.unsqueeze(-1)  # [batch_size, n_features, 1]
        # 加权融合
        fused = (stacked_features * weights).sum(dim=1)  # [batch_size, hidden_dim]        
        # 特征增强
        enhanced = self.enhance(fused) + origin
        


        return enhanced

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(258, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
class classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = KANLinear(258, 128)
            self.fc2 = KANLinear(128, 32)
            self.fc3 = KANLinear(32, 1)
            self.fc4 = KANLinear(2, 1)
            self.dropout = nn.Dropout1d(0.1)
        def forward(self, x):
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)      
            x = self.dropout(x)
            x = self.fc3(x)
            return x
        
class ProteinModel(nn.Module):
    """蛋白质序列分类的主模型
    
    整合多个功能模块，实现端到端的蛋白质序列分类。
    
    主要组件:
        - 特征处理模块：处理不同来源的输入特征
        - 特征融合模块：融合多模态特征
        - 动态分类器：自适应分类层
        - 正则化：包含dropout和权重初始化
    """
    def __init__(self):
        super().__init__()
        config = Config.MODEL_CONFIG
        MambaConfig_ = MambaConfig()
        # 只为激活的特征创建处理模块
        self.feature_blocks = nn.ModuleDict({
            name: self._create_feature_block(
                input_dim=config['input_dims'][name],
                hidden_dim=config['hidden_dims'][name],
                MambaConfig_ = MambaConfig_
            ) for name in config['fusion_config']['active_features']
        })
        # Enhanced feature fusion
        self.feature_fusion = FeatureFusion(
            input_dims=list(config['hidden_dims'].values())
        )
        
        self.features_block = BiDirectionMixerModel(
                                d_model = MambaConfig_.d_model,
                                n_layer = MambaConfig_.n_layer,
                                vocab_size = MambaConfig_.vocab_size,
                                ssm_cfg = MambaConfig_.ssm_cfg,
                                rms_norm = MambaConfig_.rms_norm,
                                residual_in_fp32 = MambaConfig_.residual_in_fp32,
                                fused_add_norm = MambaConfig_.fused_add_norm,
                                device = Config.DEVICE
                                )
        
        self.drop_path = DropPath(Config.MODEL_CONFIG['drop_path_rate']) if Config.MODEL_CONFIG['use_drop_path'] else nn.Identity()
        self._init_weights()
        self.classifier = classifier()
    def _create_feature_block(self, input_dim, hidden_dim,MambaConfig_):
        config = Config.MODEL_CONFIG
        return nn.ModuleDict({
            'conv': WeightStandardizedConv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            'norm': nn.LayerNorm(hidden_dim),
            'bimamba':BiDirectionMixerModel(
                                d_model = MambaConfig_.d_model,
                                n_layer = MambaConfig_.n_layer,
                                vocab_size = MambaConfig_.vocab_size,
                                ssm_cfg = MambaConfig_.ssm_cfg,
                                rms_norm = MambaConfig_.rms_norm,
                                residual_in_fp32 = MambaConfig_.residual_in_fp32,
                                fused_add_norm = MambaConfig_.fused_add_norm,
                                device = Config.DEVICE
                                ),
            'avgpool': MeanPooling(hidden_dim),  # 使用MeanPooling替代AttentionPooling
            'recalibration': FeatureRecalibration(hidden_dim),
            'wavelet':Wavelet(256,256)
        })
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # 使用较小的标准差进行初始化
                    nn.init.xavier_normal_(param, gain=0.5)
                else:
                    # 对于1D参数，使用较小的均匀分布
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs):
        # 只处理激活的特征
        #print("#######")
        feature = [
            self._process_feature(x, name) 
            for x, name in zip(inputs, Config.MODEL_CONFIG['input_dims'].keys())
            if name in Config.MODEL_CONFIG['fusion_config']['active_features']
        ]
        AF_data = inputs[2]
        feature = self.feature_fusion(feature)
        feature = torch.cat([feature, AF_data],dim=-1)
        feature = self.drop_path(feature)
        return torch.sigmoid(self.classifier(feature).squeeze(-1))

    def split_features_by_type(self,features):
        """
        将特征张量按特征类型拆分
        
        参数:
        features (torch.Tensor): 输入特征张量，形状为 (batch_size, feature_nums, dim)
        type_indices (list): 每种类型的特征索引列表。默认为将特征均分为两部分
        
        返回:
        tuple: 每种类型的特征张量
        """
          # 按索引提取每种类型的特征
        feature_wt, feature_vt = features[:, 0, :], features[:, 1, :]
        return feature_wt, feature_vt
    def _process_feature(self, x, name):

        block = self.feature_blocks[name]
        feature_wt, feature_vt = self.split_features_by_type(x)
        feature_wt = block['conv'](feature_wt.unsqueeze(1).transpose(1,2)).transpose(1,2)
        feature_vt = block['conv'](feature_vt.unsqueeze(1).transpose(1,2)).transpose(1,2)
        feature_wt = block['norm'](feature_wt)
        feature_vt = block['norm'](feature_vt)
        feature = block['bimamba'](feature_wt,feature_vt)
        feature = block['avgpool'](feature)
        feature = block['wavelet'](feature)

        return feature

class ExpandDim(nn.Module):
    """维度扩展模块：在指定位置添加新的维度"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)  # Expand on specified dimension


class SqueezeDim(nn.Module):
    """维度压缩模块：移除指定位置的维度"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)  # Compress specified dimension
    
class PrintShape(nn.Module):
    """形状打印模块：用于调试时查看张量形状"""
    def __init__(self, message):
        super().__init__()
        self.message = message

    def forward(self, x):
        print(f"{self.message}: {x.shape}")
        return x
    
class Transpose(nn.Module):
    """维度转置模块：交换张量的指定维度"""
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
       return x.transpose(self.dim1, self.dim2)
    
class DropPath(nn.Module):
    """随机深度正则化：在训练时随机丢弃部分路径"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            keep_prob = 1 - self.drop_prob
            mask = torch.rand(x.shape[0], 1, device=x.device) < keep_prob
            return x * mask / keep_prob
        return x

class WeightStandardizedConv1d(nn.Conv1d):
    """权重标准化的一维卷积：通过标准化权重提高模型稳定性"""
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = self.weight.mean(dim=[1,2], keepdim=True)
        var = torch.var(self.weight, dim=[1,2], keepdim=True, unbiased=False)
        normalized_weight = (self.weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)  
class ChannelAttention(nn.Module):
    """通道注意力机制：自适应调整不同通道的重要性
    
    结合平均池化和最大池化，通过全连接层学习通道权重。
    """
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.GELU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (64, 1, 256) -> (64, 256, 1)
        # print(f"[Debug] Transposed input shape: {x.shape}")  # Debug output
        
        # Average pooling and max pooling
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, channels, 1) -> (batch_size, channels)
        max_out = self.max_pool(x).squeeze(-1)  # (batch_size, channels, 1) -> (batch_size, channels)
        # print(f"[Debug] Avg pool output shape: {avg_out.shape}")  # Debug output
        # print(f"[Debug] Max pool output shape: {max_out.shape}")  # Debug output
        
        # Fully connected layer
        avg_out = self.fc(avg_out)  # (batch_size, channels)
        max_out = self.fc(max_out)  # (batch_size, channels)
        out = avg_out + max_out
        # print(f"[Debug] ChannelAttention output shape: {out.unsqueeze(-1).shape}")  # Debug output
        return x * out.unsqueeze(-1)

# class AttentionPooling(nn.Module):
#     """注意力池化层：通过学习权重进行自适应池化
    
#     使用可配置的降维比例，学习序列中每个位置的重要性。
#     """
#     def __init__(self, hidden_dim):
#         super().__init__()
#         config = Config.MODEL_CONFIG
#         mid_dim = hidden_dim // config['pooling_config']['reduction_ratio']
        
#         self.attn = nn.Sequential(
#             nn.Linear(hidden_dim, mid_dim),
#             nn.Tanh(),
#             nn.Linear(mid_dim, 1),
#             nn.Softmax(dim=1)
#         )
    
#     def forward(self, x):
#         attn_weights = self.attn(x)
#         return torch.sum(attn_weights * x, dim=1)
# 添加新的平均池化类
class Nonlinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_func1 = nn.Linear(input_dim, 768)
        self.activate_func = nn.GELU()
        self.norm_func = nn.LayerNorm(768)
        self.linear_func2 = nn.Linear(768, 256)  # 可继续堆叠更多层
    def forward(self, x):
        # 方案2：带激活函数的MLP        
        x = self.linear_func1(x)
        x = self.activate_func(x)
        x = self.norm_func(x)
        x = self.linear_func2(x)
        return x
class MeanPooling(nn.Module):
    """简单的平均池化层"""
    def __init__(self, hidden_dim):
        super().__init__()
    
    def forward(self, x):
        return x.mean(dim=1)
class MaxPooling(nn.Module):
    """简单的最大池化层"""
    def __init__(self, dim=1):
        """
        初始化最大池化层
        
        参数:
            dim (int): 沿哪个维度进行最大池化，默认为 1
        """
        super(MaxPooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        对输入张量 x 在指定维度进行最大池化
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, feature_dim)
        
        返回:
            torch.Tensor: 最大池化后的张量，形状为 (batch_size, feature_dim)
        """
        return x.max(dim=self.dim)[0]
# 添加新的ResidualBlock类
class ResidualBlock(nn.Module):
    """残差块：实现跳跃连接
    
    参数:
        main_path: 主路径的网络层序列
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, main_path, in_channels, out_channels):
        super().__init__()
        self.main_path = main_path
        # 如果输入输出维度不同，添加1x1投影层
        self.shortcut = (nn.Linear(in_channels, out_channels) 
                        if in_channels != out_channels 
                        else nn.Identity())
        
    def forward(self, x):
        identity = self.shortcut(x)
        return self.main_path(x) + identity

class FeatureRecalibration(nn.Module):
    """特征重校准模块：增强重要特征，抑制冗余特征"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 全局信息
        # x shape: [batch_size, dim]
        scale = x  # 直接使用输入，因为已经是池化后的结果
        scale = self.fc1(scale)  # [batch_size, dim//4]
        scale = F.gelu(scale)
        scale = self.fc2(scale)  # [batch_size, dim]
        scale = torch.sigmoid(scale)
        
        return self.norm(x * scale)


def get_model():
    """模型工厂函数
    
    创建并初始化完整的模型、优化器和学习率调度器。
    
    返回:
        model: 初始化好的模型
        optimizer: AdamW优化器
        scheduler: 序列化的学习率调度器(包含预热和余弦退火)
    """
    #print("######")
    model = ProteinModel().to(Config.DEVICE)
    #print("######")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.OPTIMIZER['lr'],
        weight_decay=Config.OPTIMIZER['weight_decay'],
        betas=Config.OPTIMIZER['betas'],
        eps=Config.OPTIMIZER['eps']
    )
    
    # Create Sequential LR scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            # Linear warmup scheduler
            torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                **Config.SCHEDULER['warmup']
            ),
            # Cosine annealing scheduler
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                **Config.SCHEDULER['cosine']
            )
        ],
        milestones=Config.SCHEDULER['milestones']
    )
    
    return model, optimizer, scheduler