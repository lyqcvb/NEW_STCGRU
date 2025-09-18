import os
import random
from matplotlib import axis
import scipy.io as sio
import numpy as np
import math
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import joblib
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold, train_test_split
import traingset as dl  # Ensure this module contains necessary utility functions
import logging
from mne.preprocessing import ICA
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU
seed = 42
dl.seed_everything(seed)
# EEG data parameters
duration = 1000

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
from itertools import combinations

# 初始脑区定义
regions = {
    "prefrontal": [0, 1, 2, 3, 10, 11, 16],
    "central": [4, 5, 17],
    "temporal": [12, 13, 14, 15],
    "parietal": [6, 7, 18],
    "occipital": [8, 9]
}

# 自动生成多脑区组合
def generate_combinations(regions, sizes):
    combined_regions = {}
    region_names = list(regions.keys())

    # 遍历指定组合大小
    for size in sizes:
        for combination in combinations(region_names, size):
            combined_name = "_".join(combination)  # 组合名称
            combined_indices = sorted(set().union(*(regions[region] for region in combination)))  # 合并去重
            combined_regions[combined_name] = combined_indices

    return combined_regions

# 生成所有二、三、四脑区组合
regions = generate_combinations(regions, sizes=[1,2, 3, 4,5])
# 动态获取变量值
partition = "prefrontal_central_temporal_parietal_occipital"

def importAndCropData(file_set, duration, labels,partition):
    EEG_list = []
    label = []
    name = []
    for i, file in enumerate(file_set):
        try:
            raw = mne.io.read_raw_edf(file[0], preload=True, encoding='latin1',verbose='Warning')
            local_name = file[1]
            data = raw.get_data()[0:19]
            data = data[partition]
            if data.shape[1] > duration:
                epochs = data.shape[1] // duration
                data_crop = data[:,0:epochs*duration]
            else:
                continue
            label += [labels[i]] * epochs
            name  += [local_name] * epochs
            channels = len(partition)
            data_new = data_crop.reshape(channels, -1, duration).transpose(1, 0, 2)
            EEG_list.append(data_new)
            logging.info(f"Processed file {file}: {epochs} epochs")
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            continue

    if not EEG_list:
        raise ValueError("No data was loaded. Please check the file paths and formats.")
    
    EEG = np.concatenate(EEG_list)
    label = np.array(label)
    name = np.array(name)
    logging.info(f"Total epochs: {EEG.shape[0]}, Normal: {np.sum(label == 1)}, "
            f"MCI: {np.sum(label == 0)}")
    return EEG,label,name

import os
import warnings
# 忽略 RuntimeWarning 警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 定义文件夹路径
base_dir = '糖尿病认知障碍与对照脑电数据'
normal_dir = os.path.join(base_dir, '认知正常')
impaired_dir = os.path.join(base_dir, '认知障碍')

# 获取所有的文件路径
normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.edf')]
impaired_files = [os.path.join(impaired_dir, f) for f in os.listdir(impaired_dir) if f.endswith('.edf')]

# 获取所有的文件路径和文件名（以二元组存储）
normal_files = [(os.path.join(normal_dir, f), f[:-4]) for f in os.listdir(normal_dir) if f.endswith('.edf')]
impaired_files = [(os.path.join(impaired_dir, f), f[:-4]) for f in os.listdir(impaired_dir) if f.endswith('.edf')]

all_files = normal_files + impaired_files
label_single = np.concatenate([np.ones(len(impaired_files)), np.zeros(len(normal_files))],axis=0)
# 将 all_files 和 label_single 中的元素按相同顺序打乱
combined = list(zip(all_files, label_single))
random.shuffle(combined)
all_files[:], label_single[:] = zip(*combined)
original_data,labels,name = importAndCropData(all_files, duration, label_single,regions[partition])
final_data =original_data.reshape(original_data.shape[0],1,original_data.shape[1],original_data.shape[2])
print(len(original_data))

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Ensure output directories exist
ensure_dir("EEGData/"+str(partition)+"/TrainData")
ensure_dir("EEGData/"+str(partition)+"/ValidData")
ensure_dir("EEGData/"+str(partition)+"/TestData")

# 假设 final_data, labels 是已有的 numpy 数组
# final_data: shape = (samples, 时间长度)
# labels: shape = (samples, 1)，每个样本的标签为 0 或 1

# 将 labels 从二维转换为一维
labels = labels.reshape(-1)

# 创建 StratifiedKFold 对象，指定 10 折交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# 遍历每一折
for fold, (train_idx, val_idx) in enumerate(kf.split(final_data, labels)):
    try:
        train_data, test_data = final_data[train_idx], final_data[val_idx]
        train_labels, test_labels = labels[train_idx], labels[val_idx]
        
        train_data_split, valid_data_split, train_labels_split, valid_labels_split = train_test_split(
                train_data, train_labels, test_size=0.1, random_state=seed, stratify=train_labels
            )
        # print(train_data_split.shape,train_labels_split.shape,valid_data_split.shape,valid_labels_split.shape)
        # Convert to PyTorch tensors
        train_tensor = torch.from_numpy(train_data_split).float() # (samples, channels, duration)
        train_labels_tensor = torch.from_numpy(train_labels_split).long()

        valid_tensor = torch.from_numpy(valid_data_split).float()
        valid_labels_tensor = torch.from_numpy(valid_labels_split).long()

        test_tensor = torch.from_numpy(test_data).float()
        test_labels_tensor = torch.from_numpy(test_labels).long()

        # Create TensorDatasets
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        valid_dataset = TensorDataset(valid_tensor, valid_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)

        # Save datasets
        torch.save(train_dataset, "EEGData/"+str(partition)+f"/TrainData/train_data_{fold + 1}_fold_with_seed_{seed}.pth")
        torch.save(valid_dataset, "EEGData/"+str(partition)+f"/ValidData/valid_data_{fold + 1}_fold_with_seed_{seed}.pth")
        torch.save(test_dataset, "EEGData/"+str(partition)+f"/TestData/test_data_{fold + 1}_fold_with_seed_{seed}.pth")

        logging.info(f"Fold {fold + 1} data saved successfully.")
            # 转换 y_train 和 y_val 为整数类型
        y_train = train_labels_split.astype(int)
        y_val = valid_labels_split.astype(int)
        y_test = test_labels.astype(int)
        # 输出当前折的训练集和验证集和测试集大小
        print(f"Fold {fold+1}:")
        print(f"  训练集大小: {y_train.shape}, 验证集大小: {y_val.shape}, 测试集大小: {y_test.shape}")
        print(f"  训练集标签分布: {np.bincount(y_train)}")
        print(f"  验证集标签分布: {np.bincount(y_val)}")
        print(f"  测试集标签分布: {np.bincount(y_test)}\n")
    except Exception as e:
        logging.error(f"Error processing fold {fold + 1}: {e}")

print("存储区:",partition)
