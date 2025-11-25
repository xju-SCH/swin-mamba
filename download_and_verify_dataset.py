#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载并验证COCO数据集子集是否满足Swin Transformer的要求
原始尺寸要求：512K及以上
"""

import os
import sys
import requests
import zipfile
import shutil
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

# 设置中文字符支持
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class COCOSubsetDownloader:
    def __init__(self, save_dir="/root/autodl-fs/vim-shiyan/data/coco_subset", max_images=50):
        self.save_dir = save_dir
        self.max_images = max_images
        self.images_dir = os.path.join(save_dir, "images")
        self.annotations_file = os.path.join(save_dir, "annotations.json")
        # COCO验证集的一小部分示例URL（1GB左右）
        self.dataset_url = "http://images.cocodataset.org/zips/val2017.zip"
        self.annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        # 创建保存目录
        os.makedirs(self.images_dir, exist_ok=True)
    
    def download_file(self, url, save_path):
        """下载文件并显示进度"""
        print(f"正在下载: {url}")
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            print(f"文件已存在: {save_path}")
            return True
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            
            with open(save_path, 'wb') as file, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            
            print(f"下载完成: {save_path}")
            return True
        except Exception as e:
            print(f"下载失败: {str(e)}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """解压ZIP文件"""
        print(f"正在解压: {zip_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"解压完成: {extract_to}")
            return True
        except Exception as e:
            print(f"解压失败: {str(e)}")
            return False
    
    def create_subset(self):
        """创建COCO数据集的小型子集"""
        # 下载并解压验证集图像
        images_zip = os.path.join(self.save_dir, "val2017.zip")
        if not self.download_file(self.dataset_url, images_zip):
            print("尝试使用其他较小的数据集选项...")
            # 如果无法下载完整验证集，尝试使用更简单的方法
            return self.download_sample_images()
        
        temp_extract_dir = os.path.join(self.save_dir, "temp")
        self.extract_zip(images_zip, temp_extract_dir)
        
        # 移动部分图像到子集目录
        original_images_dir = os.path.join(temp_extract_dir, "val2017")
        if os.path.exists(original_images_dir):
            image_files = [f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg'))]
            print(f"找到 {len(image_files)} 张图像，选择前 {self.max_images} 张")
            
            # 选择满足大小要求的图像
            selected_images = []
            for img_file in image_files[:self.max_images * 2]:  # 多选一些以防有些不满足大小要求
                img_path = os.path.join(original_images_dir, img_file)
                img_size_kb = os.path.getsize(img_path) / 1024
                if img_size_kb >= 512:  # 512KB及以上
                    shutil.copy(img_path, os.path.join(self.images_dir, img_file))
                    selected_images.append(img_file)
                    print(f"选择图像: {img_file}, 大小: {img_size_kb:.2f}KB")
                    if len(selected_images) >= self.max_images:
                        break
            
            print(f"已选择 {len(selected_images)} 张满足大小要求的图像")
            
            # 清理临时文件
            shutil.rmtree(temp_extract_dir)
            return len(selected_images) > 0
        
        return False
    
    def download_sample_images(self):
        """下载一些示例高分辨率图像作为备选方案"""
        print("使用备选方案：下载示例高分辨率图像")
        
        # 创建一个小型的JSON注释文件
        annotations = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 2, "name": "car", "supercategory": "vehicle"},
                {"id": 3, "name": "dog", "supercategory": "animal"}
            ]
        }
        
        # 模拟一些高分辨率图像的信息
        sample_image_info = [
            {"id": 1, "file_name": "sample_1.jpg", "width": 1024, "height": 768, "size_kb": 850},
            {"id": 2, "file_name": "sample_2.jpg", "width": 1200, "height": 800, "size_kb": 950},
            {"id": 3, "file_name": "sample_3.jpg", "width": 1920, "height": 1080, "size_kb": 1200},
        ]
        
        # 创建模拟的高分辨率图像
        for info in sample_image_info:
            img_path = os.path.join(self.images_dir, info["file_name"])
            if not os.path.exists(img_path):
                # 创建一个随机高分辨率图像
                img_array = np.random.randint(0, 256, (info["height"], info["width"], 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(img_path, quality=85)
                print(f"创建模拟高分辨率图像: {info['file_name']}, 尺寸: {info['width']}x{info['height']}, 估计大小: {info['size_kb']}KB")
            
            annotations["images"].append({
                "id": info["id"],
                "file_name": info["file_name"],
                "width": info["width"],
                "height": info["height"]
            })
        
        # 保存注释文件
        with open(self.annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        return True
    
    def verify_dataset(self):
        """验证数据集是否满足Swin Transformer的要求"""
        print("\n验证数据集...")
        
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg'))]
        if not image_files:
            print("错误：未找到任何图像文件")
            return False
        
        # 统计信息
        total_images = len(image_files)
        valid_images = 0
        total_size_kb = 0
        min_resolution = (float('inf'), float('inf'))
        max_resolution = (0, 0)
        
        for img_file in image_files:
            img_path = os.path.join(self.images_dir, img_file)
            
            try:
                # 检查文件大小
                file_size_kb = os.path.getsize(img_path) / 1024
                total_size_kb += file_size_kb
                
                # 检查图像分辨率
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                # 验证是否满足要求
                if file_size_kb >= 512:
                    valid_images += 1
                    min_resolution = (min(min_resolution[0], width), min(min_resolution[1], height))
                    max_resolution = (max(max_resolution[0], width), max(max_resolution[1], height))
                    print(f"图像: {img_file}, 尺寸: {width}x{height}, 大小: {file_size_kb:.2f}KB ✓")
                else:
                    print(f"图像: {img_file}, 尺寸: {width}x{height}, 大小: {file_size_kb:.2f}KB ✗ (小于512KB)")
                    
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {str(e)}")
        
        # 计算平均大小和分辨率
        avg_size_kb = total_size_kb / total_images if total_images > 0 else 0
        
        print("\n数据集验证结果:")
        print(f"总图像数: {total_images}")
        print(f"有效图像数 (≥512KB): {valid_images}")
        print(f"平均图像大小: {avg_size_kb:.2f}KB")
        print(f"最小分辨率: {min_resolution[0]}x{min_resolution[1]}")
        print(f"最大分辨率: {max_resolution[0]}x{max_resolution[1]}")
        print(f"总数据集大小: {total_size_kb/1024:.2f}MB")
        
        # 验证是否满足Swin Transformer的要求
        if valid_images >= 10:
            print("\n✓ 数据集满足基本要求")
            print("\n使用说明：")
            print(f"1. 数据集位置: {self.save_dir}")
            print("2. 可以修改Swin Transformer配置文件中的以下参数:")
            print("   - DATA.DATA_PATH: 设置为 'data/coco_subset/images'")
            print("   - DATA.IMG_SIZE: 根据需要设置，建议值包括 512 或 384")
            return True
        else:
            print("\n✗ 数据集不满足要求，请增加有效图像数量")
            return False

def verify_swin_transformer_compatibility():
    """验证数据集与Swin Transformer的兼容性"""
    print("\n验证与Swin Transformer的兼容性...")
    
    # 模拟Swin Transformer的图像预处理
    def simulate_swin_preprocessing(image_size=512):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform
    
    # 测试不同尺寸的兼容性
    test_sizes = [384, 512, 640]
    print("Swin Transformer兼容性测试:")
    
    for size in test_sizes:
        print(f"- 支持图像尺寸: {size}x{size}")
        # 计算与patch大小的兼容性
        patch_size = 4  # Swin Transformer默认patch大小
        if size % patch_size == 0:
            print(f"  ✓ {size}x{size} 与patch大小 {patch_size} 兼容 (可整除)")
        else:
            print(f"  ! {size}x{size} 与patch大小 {patch_size} 不完全兼容 (不可整除)")
    
    print("\n兼容性建议:")
    print("1. 对于高分辨率图像，建议使用较大的窗口大小(window_size)")
    print("2. 较大的图像尺寸可能需要更多的GPU内存")
    print("3. 可以通过修改配置文件中的 MODEL.SWIN.WINDOW_SIZE 来优化性能")
    print("4. 对于512x512图像，推荐使用window_size=16或更大")
    
    return True

def main():
    print("=== COCO数据集子集下载与验证工具 ===")
    print("目标: 创建满足Swin Transformer要求的小型高分辨率数据集")
    print("要求: 原始图像尺寸≥512KB\n")
    
    # 创建下载器实例
    downloader = COCOSubsetDownloader(max_images=20)
    
    # 创建数据集子集
    if not downloader.create_subset():
        print("警告：未能创建完整的COCO子集，使用模拟高分辨率图像")
    
    # 验证数据集
    is_valid = downloader.verify_dataset()
    
    # 验证与Swin Transformer的兼容性
    verify_swin_transformer_compatibility()
    
    # 生成使用示例命令
    print("\n=== 使用示例 ===")
    print("运行Swin Transformer训练命令示例:")
    print("python -m torch.distributed.launch --nproc_per_node=1 ")
    print("    --master_port=12345 main.py ")
    print(f"    --cfg configs/swin/swin_base_patch4_window12_384_finetune.yaml ")
    print(f"    --data-path {downloader.images_dir} ")
    print("    --img-size 512 ")
    print("    --batch-size 2 ")
    print("    --output output_coco_subset ")
    print("    --tag coco_subset_512")
    
    if is_valid:
        print("\n✓ 数据集准备完成，可以用于Swin Transformer")
    else:
        print("\n! 数据集需要进一步调整以满足要求")

if __name__ == "__main__":
    main()