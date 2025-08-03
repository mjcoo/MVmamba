import torch

class Config:
    # 设备配置
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    NUM_WORKERS = 4

    # 数据路径
    DATA_PATH = './dataset/train'
    TEST_DATA_PATH = './dataset/test'
    MODEL_PATH = './model'
    LOG_PATH = './logs'
    CSV_PATH = './logs/fold_metrics'
    RESULT_PATH = './result'
    TEST_RESULT_PATH = './test_result'

    # 特征配置
    FEATURE_CONFIG = {
        'active_features': ['esmc_pdb_global', 'esmc_pdb_local'],

        'feature_info': {
            'esmc_pdb_local': {'dim': 1152, 'path': 'windows/train_esmc_pdb_local_window1.npy'},
            'esmc_pdb_global': {'dim': 1152, 'path': 'pdb/train_esmc_pdb_global.npy'},
            'esm3_pdb_local': {'dim': 1536, 'path': 'train_esm3_pdb_local.npy'},
            'esm3_pdb_global': {'dim': 1536, 'path': 'train_esm3_pdb_global.npy'},
        }
    }
    TEST_FEATURE_CONFIG = {
        'active_features': ['esmc_pdb_global', 'esmc_pdb_local'],

        'feature_info': {
            'esmc_pdb_local': {'dim': 1152, 'path': 'windows/test_esmc_pdb_local_window1.npy'},
            'esmc_pdb_global': {'dim': 1152, 'path': 'pdb/test_esmc_pdb_global.npy'},
            'esm3_pdb_local': {'dim': 1536, 'path': 'train_esm3_pdb_local.npy'},
            'esm3_pdb_global': {'dim': 1536, 'path': 'train_esm3_pdb_global.npy'},
        }
    }
    mmmamba = {
            'mm_input_size': 256,
            'mm_output_sizes': [256],
            'dropout': 0.1,
            'd_ffn': 1024 ,
            'num_layers': 1,
            'activation': 'GELU',
            'causal': 'false',

            'mamba_config':{
                'd_state': 12 ,
                'expand': 4 ,
                'd_conv': 4 ,
                'bidirectional': 'true'
            }
    }
    # 模型参数优化
    MODEL_CONFIG = {
        # 降低模型复杂度，避免过拟合

        'hidden_dims': {
            'esm1b_local': 256,          # 降低维度
            'esm1b_global': 256,           
            'esmc_local': 256,  
            'esmc_global': 256, 
            'esmc_pdb_local': 256,    
            'esmc_pdb_global': 256,
            'esm3_local': 256,  
            'esm3_global': 256, 
            'esm3_pdb_local': 256,    
            'esm3_pdb_global': 256,    
            'protT5local': 256,    
            'protT5global': 256,   
            'structure_score': 256  # 统一维度
        },

        # 简化编码器
        'encoder_config': {
            'num_layers': 2,        # 减少层数
            'dropout_rate': 0.1,    # 降低dropout
        },

        # 简化注意力
        'pooling_config': {
            'reduction_ratio': 4
        },

        # 简化特征融合
        'fusion_config': {
            'common_dim': 256,      # 统一维度
            'attention_hidden_dim': 64,
            'cross_attention_pairs': 'all',
            'fusion_type': 'cross_attention'
        },

        # 简化分类器
        'classifier_config': {
            'hidden_dims': [128, 64],  # 简化网络
            'dropouts': [0.1, 0.1]     # 降低dropout
        },

        'use_drop_path': True,
        'drop_path_rate': 0.1,     # 降低drop path率
    }

    # Add the input_dims and active_features after class definition
    @classmethod
    def initialize(cls):
        # Add input_dims to MODEL_CONFIG
        cls.MODEL_CONFIG['input_dims'] = {
            name: cls.FEATURE_CONFIG['feature_info'][name]['dim']
            for name in cls.FEATURE_CONFIG['active_features']
        }
        
        # Add active_features to fusion_config
        cls.MODEL_CONFIG['fusion_config']['active_features'] = cls.FEATURE_CONFIG['active_features']

    # 训练参数优化
    BATCH_SIZE = 512  # 减小batch size
    TRAIN_VAL_SPLIT = 0.8
    MAX_EPOCHS = 200
    PATIENCE = 5

    # Add cross validation config
    CV_CONFIG = {
        'n_splits': 10,  # 5-fold cross validation
        'shuffle': True
    }
    
    # Add metrics file path
    METRICS_FILE = 'fold_metrics_{timestamp}.csv'

    # 优化器参数调整
    OPTIMIZER = {
        'lr': 1e-4,              # 降低学习率
        'weight_decay': 0.001,   # 降低权重衰减
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }

    # 学习率调度优化
    SCHEDULER = {
        'warmup': {
            'start_factor': 0.3,   # 提高起始学习率
            'end_factor': 1.0,
            'total_iters': 1       # 减少预热
        },
        'cosine': {
            'T_0': 10,            
            'T_mult': 2,          
            'eta_min': 1e-6
        },
        'milestones': [1]
    }

    # 损失函数优化
    FOCAL_LOSS = {
        'gamma': 0.5,    # 降低gamma，减少对难样本的关注
        'alpha': 0.5     # 平衡正负样本
    }

    LOSS = {
        'l1_lambda': 1e-6,       # 显著降低L1正则化
        'label_smoothing': 0.05  # 减少标签平滑
    }

# Initialize the dynamic configurations
Config.initialize()