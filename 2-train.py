import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import joblib

import traingset 
import resnet
import STCGRU 
import RESGRU
import PSDCnn

# 定义脑区索引
from itertools import combinations

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

# 初始脑区定义
regions = {
    "prefrontal": [0, 1, 2, 3, 10, 11, 16],
    "central": [4, 5, 17],
    "temporal": [12, 13, 14, 15],
    "parietal": [6, 7, 18],
    "occipital": [8, 9]
}

# 生成所有二、三、四脑区组合
regions = generate_combinations(regions, sizes=[1,2, 3, 4,5])


# 动态获取变量值
seed = 42
total_fold = 10  # 10折
'''深度学习超参数'''
num_classes = 2
batch_size = 64
num_epochs = 50
learning_rate = 0.0001
traingset.seed_everything(seed)
partition = "prefrontal"
model_name = "RESGRU_V1_tnb"
dataset_name = "EEGData"
writer = SummaryWriter('./runs/' +model_name+'/'+partition+"_"+str(seed))
num_channels = len(regions[partition])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

start = time.perf_counter()
for i in range(total_fold):
    train_data_combine = torch.load(dataset_name+"/"+partition+"/TrainData/train_data_"
                                    + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth",weights_only=False)
    valid_data_combine = torch.load(dataset_name+"/"+partition+"/ValidData/valid_data_"
                                    + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth",weights_only=False)
    '''定义深度学习模型'''
    # model = STCGRU.model(tunnelNums=num_channels).to(device)
    model = RESGRU.AblationModel_SingleResBlock().to(device)
    # model = PSDCnn.PSDNet(input_channels=num_channels).to(device)
    '''定义损失函数Loss 和 优化算法optimizer'''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    # In 2-train.py
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,      # 绑定的优化器
    #     mode='min',     # 'min' 模式，监控指标越小越好 (例如 loss)
    #                     # 'max' 模式，监控指标越大越好 (例如 accuracy)
    #     factor=0.1,     # 学习率衰减系数。新学习率 = 旧学习率 * factor
    #     patience=10,    # “耐心值”，即能容忍多少个epoch内指标不优化
    #     verbose=True    # 是否在调整学习率时在控制台打印提示信息
    # )
    print('开始第%d次训练，共%d次' % (i + 1, total_fold))
    # 生成迭代器，根据小批量数据大小划分每批送入模型的数据集
    train_loader = DataLoader(dataset=train_data_combine,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=0)
    valid_loader = DataLoader(dataset=valid_data_combine,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=0)
    total_step = len(train_loader)
    '''模型训练'''
    for epoch in range(num_epochs):
        '''训练'''
        traingset.model_training(writer, i, type='train', num_epochs=num_epochs,
                                            epoch=epoch, loader=train_loader, model=model,
                                            criterion=criterion, optimizer=optimizer)
        '''验证'''
        traingset.model_training(writer, i, type='validation', epoch=epoch,
                                            loader=valid_loader, model=model, criterion=criterion,
                                            optimizer=optimizer, scheduler=scheduler)
    ensure_dir(model_name+"/"+partition)
    torch.save(model.state_dict(),model_name+"/"+partition + "/"+ str(i + 1) + "_fold_model_parameter_with_seed_" + str(seed) + ".pth")
    print(model_name + "模型第" + str(i + 1) + "次训练结果保存成功")
end = time.perf_counter()
print("训练及验证运行时间为", round(end - start), 'seconds')