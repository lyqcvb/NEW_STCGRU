import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import joblib
import model as NN

seed = 22
total_fold = 10  # 10折
'''深度学习超参数'''
input_size = 16
hidden_size = 128
num_layers_lstm = 1
num_layers_bilstm = 2
num_classes = 2
batch_size = 40
num_epochs = 50
# learning_rate = 0.0003
learning_rate = 0.001

start = time.perf_counter()
NN.seed_everything(seed)

# 定义脑区索引
regions = {
    "prefrontal": [0, 1, 2, 3, 10, 11, 16],
    "central": [4, 5, 17],
    "temporal": [12, 13, 14, 15],
    "parietal": [6, 7, 18],
    "occipital": [8, 9],
    "all":[]
}

# 动态获取变量值
partition = "temporal"


srate ="32"
writer = SummaryWriter('./runs/' +partition+"_"+str(seed))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

for i in range(total_fold):
    train_data_combine = torch.load("EEGData/"+partition+"/TrainData/train_data_"
                                    + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth",weights_only=False)
    valid_data_combine = torch.load("EEGData/"+partition+"/ValidData/valid_data_"
                                    + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth",weights_only=False)
    '''定义深度学习模型'''
    model = NN.STCGRU().to(device)
    '''定义损失函数Loss 和 优化算法optimizer'''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)  # 余弦退火
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
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
        model, optimizer = NN.model_training(writer, i, type='train', num_epochs=num_epochs,
                                            epoch=epoch, loader=train_loader, neural_network=model,
                                            criterion=criterion, optimizer=optimizer)
        '''验证'''
        optimizer, lr_list = NN.model_training(writer, i, type='validation', epoch=epoch,
                                            loader=valid_loader, neural_network=model, criterion=criterion,
                                            optimizer=optimizer, scheduler=scheduler)
    ensure_dir("stcgru/"+partition)
    torch.save(model.state_dict(),
            "stcgru/"+partition + "/"+ str(i + 1) + "_fold_model_parameter_with_seed_" + str(seed) + ".pth")
    print("stcgru" + "模型第" + str(i + 1) + "次训练结果保存成功")
end = time.perf_counter()
print("训练及验证运行时间为", round(end - start), 'seconds')