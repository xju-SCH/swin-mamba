# Swin Mamba (直接集成Mamba实现)

本项目是通过将Mamba SSM直接集成到Swin Transformer架构中实现的Swin Mamba。

## 项目结构

- `models/`: 存放模型实现
  - `swin_mamba.py`: Swin Mamba的核心实现
  - `mamba_ssm/`: 直接复制的Mamba SSM实现
- `configs/`: 配置文件
- `data/`: 数据处理相关
- `main.py`: 训练脚本

## 实现方式

本项目直接将mamba-1p1p1文件夹复制到项目中，然后引用其中的Mamba模块来替换Swin Transformer中的WindowAttention。
