
from torch.backends import cudnn
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import numpy as np
import os
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True


def model_training(tensorboard, i, type, loader, model, criterion, epoch=None, num_epochs=None, optimizer=None,scheduler=None):
    # ==================== 核心修正 ====================
    # 1. 在函数开始时，根据类型明确设置模型模式
    # if type == 'train':
    #     model.train()
    # else:
    #     model.eval()
    # ================================================

    writer = tensorboard
    metric = Accumulator(3)  # (累加总损失, 累加正确预测数, 累加总样本数)
    
    # 使用上下文管理器来优雅地控制梯度计算，这样更安全
    with torch.set_grad_enabled(type == 'train'):
        for step, (data, labels) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss_step = criterion(outputs, labels)

            # 只在训练模式下执行反向传播和优化
            if type == 'train':
                optimizer.zero_grad()
                loss_step.mean().backward()
                optimizer.step()

            # 修正：累加 (loss * 批次大小)，而不是只累加loss
            metric.add(loss_step.item() * data.size(0), accuracy(outputs, labels), data.size(0))

    # --- 后续的打印和返回逻辑 ---
    # 增加一个保护，防止loader为空时分母为0
    total_samples = metric[2] if metric[2] > 0 else 1
    avg_loss = metric[0] / total_samples
    avg_acc = metric[1] / total_samples

    if type == 'train':
        writer.add_scalars(type + '_loss', {type + str(i): avg_loss}, epoch + 1)
        writer.add_scalars(type + '_accuracy', {type + str(i): avg_acc}, epoch + 1)
        # 使用 f-string 让打印更简洁，并用 end="" 避免自动换行，方便后续打印验证结果
        print(f"Epoch: [{epoch + 1:3d}/{num_epochs}] Train loss: {avg_loss:.4f}  Train accuracy: {avg_acc:.4f}", end="")
        return
    
    if type == 'validation':
        # 只有在验证时才调用 scheduler.step()
        # 注意: 如果是 ReduceLROnPlateau，需要传入验证损失
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        writer.add_scalars(type + '_loss', {type + str(i): avg_loss}, epoch + 1)
        writer.add_scalars(type + '_accuracy', {type + str(i): avg_acc}, epoch + 1)
        print(f"  |  Validation loss: {avg_loss:.4f}  Validation accuracy: {avg_acc:.4f}")
        return

def model_predict(i, loader, model):
    # model.eval()  # 切换到评估模式
    acc = 0
    results_sum = []
    labels_test_sum = []
    results_PR_sum = []

    with torch.no_grad(): # 关闭梯度计算
        for step, (data, labels) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            acc += outputs.argmax(dim=1).eq(labels).float().mean().item()
            results_sum.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_test_sum.extend(labels.cpu().numpy())
            results_PR_sum.extend(outputs.detach().cpu().numpy())

    acc_average = acc / (step + 1)
    return acc_average, results_sum, labels_test_sum, results_PR_sum
