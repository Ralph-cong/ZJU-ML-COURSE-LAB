import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition

from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard

def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return:
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)

    return train_data_loader, valid_data_loader

# 数据集路径
data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/image"
train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=8)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

epochs = 100
model = MobileNetV1(classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'max',
                                                 factor=0.8,
                                                 patience=2)
# 损失函数
criterion = nn.CrossEntropyLoss()

# 初始化 TensorBoard 记录器
writer = SummaryWriter(log_dir='./runs/experiment256_100_batch8')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (x, y) in enumerate(train_data_loader, 1):
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)

        # print(pred_y.shape)
        # print(y.shape)

        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # 每个批次记录到 TensorBoard
        writer.add_scalar('Batch Loss', loss.item(), epoch * len(train_data_loader) + batch_idx)

    # 记录每个 epoch 的平均损失
    avg_loss = epoch_loss / len(train_data_loader)
    writer.add_scalar('Epoch Loss', avg_loss, epoch)

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for x_val, y_val in valid_data_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            pred_y_val = model(x_val)

            val_loss += criterion(pred_y_val, y_val).item()
            _, predicted = torch.max(pred_y_val, 1)
            correct += (predicted == y_val).sum().item()
            total += y_val.size(0)

    avg_val_loss = val_loss / len(valid_data_loader)
    val_accuracy = correct / total

    # 记录验证损失和准确率
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    # 调用学习率调度器
    scheduler.step(avg_val_loss)

    print(f'Epoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

torch.save(model.state_dict(), './results/temp.pth')
print('Finish Training.')
writer.close()