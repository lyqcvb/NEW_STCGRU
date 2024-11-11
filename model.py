from sympy import false
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Split_Sets(total_fold, data):
    # total_fold是你设定的几折，我这里之后带实参带10就行，data就是我需要划分的数据
    # train_index,test_index用来存储train和test的index（索引）
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
    # 这里设置shuffle设置为ture就是打乱顺序在分配
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    from torch.backends import cudnn
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True

def model_training(tensorboard, i, type, loader, neural_network, criterion, epoch=None, num_epochs=None, optimizer=None,
                scheduler=None):
    writer = tensorboard
    name = locals()
    loss = 0
    acc = 0
    lr_list = []
    results_sum = []
    labels_test_sum = []
    results_PR_sum = []
    for step, (data, labels) in enumerate(loader):
        data = data.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = neural_network(data)
        loss_step = criterion(outputs, labels)
        if type == 'train':
            # 反向传播
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
        acc += outputs.argmax(dim=1).eq(labels).type_as(torch.FloatTensor()).mean()
        loss += loss_step.item()
        if type == 'test':
            results_sum = np.append(results_sum, outputs.argmax(dim=1).cpu().numpy())
            labels_test_sum = np.append(labels_test_sum, labels.cpu().numpy())
            results_PR_sum.extend(outputs.detach().cpu().numpy())
    acc_average = acc / (step + 1)

    loss_average = loss / (step + 1)
    if type == 'train':
        writer.add_scalars(type + '_loss', {type + str(i): loss_average}, epoch + 1)
        writer.add_scalars(type + '_accuracy', {type + str(i): acc_average}, epoch + 1)
        # os.system('nvidia-smi')
        print("Epoch: [{:3d}/{}]".format(epoch + 1, num_epochs),
            "Train loss: {:.4f}".format(loss_average),
            "     "
            "Train accuracy: {:.4f}".format(acc_average))
        return neural_network, optimizer
    if type == 'validation':
        scheduler.step()
        lr_list.append(optimizer.param_groups[0]['lr'])
        writer.add_scalars(type + '_loss', {type + str(i): loss_average}, epoch + 1)
        writer.add_scalars(type + '_accuracy', {type + str(i): acc_average}, epoch + 1)
        print("                 "
            "Validation loss: {:.4f}".format(loss_average),
            "Validation accuracy: {:.4f}".format(acc_average))
        return optimizer, lr_list
    if type == 'test':
        print("第" + str(i + 1) + "次训练测试集准确率: {:.4f}".format(acc_average))
        return acc_average, results_sum, labels_test_sum, results_PR_sum
# device = torch.device('cpu')
class STCGRU(nn.Module):
    def __init__(self):
        super(STCGRU, self).__init__()
        # 大卷积层
        self.cnn_layer_1_large = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1, 77), stride=(1,3),bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1, 39), stride=(1,3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # 小卷积层
        self.cnn_layer_1_small = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1, 21), stride=(1,3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1,11), stride=(1,3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # 一维卷积层 SCNN
        self.cnn_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=19, out_channels=19,
                kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=19),
            nn.ReLU(),
            nn.Conv1d(in_channels=19, out_channels=38,
                kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=38),
            nn.ReLU(),
            nn.Conv1d(in_channels=38, out_channels=76,
                    kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=76),
            nn.ReLU(),
        )

        self.fc_layer_1 = nn.Linear(128, 128)
        self.gru = nn.GRU(input_size=76, hidden_size=32,
                        num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc_layer_2 = nn.Linear(32 * 2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过两个卷积层
        out1 = self.cnn_layer_1_large(torch.unsqueeze(x, dim=1))
        out2 = self.cnn_layer_1_small(torch.unsqueeze(x, dim=1))

        # 确保尺寸匹配
        # print("out1 shape:", out1.shape)  # 检查 out1 形状
        # print("out2 shape:", out2.shape)  # 检查 out2 形状

        # 合并输出
        out = torch.cat((out1, out2), dim=3)

        # 改变维度以匹配 cnn_layer_2 的输入
        out = out.view(out.size(0), out.size(2), -1)
        # print("out shape:", out.shape)  # 检查 out 形状
        out = self.cnn_layer_2(out)
        # print("out shape:", out.shape)  # 检查 out 形状
        # 全连接层和GRU
        out = self.fc_layer_1(out)
        # print("out shape:", out.shape)  # 检查 out 形状
        out = out.permute(0, 2, 1)
        h0 = torch.zeros(1 * 2, x.size(0), 32).to(device)
        # print("out shape:", out.shape)  # 检查 out 形状
        # print("h0 shape:", h0.shape)  # 检查 h0 形状    
        # GRU 层
        _, hn = self.gru(out, h0)
        # print("hn shape:", hn.shape)
        # 展平和输出
        out = self.flatten(hn.permute(1, 0, 2))
        out = self.fc_layer_2(out)
        out = self.sigmoid(out)
        return out



