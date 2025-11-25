
import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import SwinMamba

def parse_args():
    parser = argparse.ArgumentParser(description='Swin Mamba 训练脚本')
    parser.add_argument('--data_path', default='/path/to/dataset', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_classes', type=int, default=1000, help='分类数量')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--save_path', default='./output', help='模型保存路径')
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建模型
    model = SwinMamba(
        img_size=args.img_size,
        patch_size=4,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        d_state=16,
        rms_norm=False
    )
    model = model.to(device)

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印训练信息
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        # 更新学习率
        scheduler.step()

        # 保存模型
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }, os.path.join(args.save_path, f'swin_mamba_epoch_{epoch+1}.pth'))

    print('训练完成!')

if __name__ == '__main__':
    main()
