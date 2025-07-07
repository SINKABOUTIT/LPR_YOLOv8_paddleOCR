import os
import shutil
from PIL import Image
import random

# --- 配置参数 ---
# CCPD2019 base 数据集路径，请根据你的实际路径修改
CCPD_BASE_DIR = 'your_location'
# 输出YOLOv8格式数据集的根目录
OUTPUT_YOLO_DATASET_DIR = 'Your_location'
# 训练集、验证集、测试集比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 类别信息（车牌检测通常只有一个类别：车牌）
CLASS_NAME = 'license_plate'
CLASS_ID = 0  # YOLOv8的类别ID从0开始


# --- 函数定义 ---

def create_yolo_dataset_structure(output_dir):
    """创建YOLOv8所需的数据集目录结构"""
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)
    print(f"Created dataset structure at: {output_dir}")


def parse_ccpd_filename(filename, img_width, img_height):



    parts = filename.split('-')
    if len(parts) < 3:
        print(f"Warning: Filename format error for {filename}. Skipping.")
        return None

    bbox_str = parts[2]
    try:
        # 分割出所有 x&y 形式的字符串
        point_strs = bbox_str.split('_')

        all_x = []
        all_y = []

        for p_str in point_strs:
            if '&' in p_str:
                x_str, y_str = p_str.split('&')
                all_x.append(int(x_str))
                all_y.append(int(y_str))

        if not all_x or not all_y:
            print(f"Warning: Could not parse points from {filename}. Skipping.")
            return None

        min_x = min(all_x)
        min_y = min(all_y)
        max_x = max(all_x)
        max_y = max(all_y)

        # 计算YOLO格式的中心点和宽高
        box_width = max_x - min_x
        box_height = max_y - min_y
        x_center = min_x + box_width / 2
        y_center = min_y + box_height / 2

        # 归一化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = box_width / img_width
        height_norm = box_height / img_height

        # 确保归一化值在0-1之间，并处理可能的微小越界
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        # 确保宽高不是0（如果图像尺寸太小或标注有问题）
        if width_norm == 0 or height_norm == 0:
            print(f"Warning: Zero width or height detected for {filename}. Skipping.")
            return None

        return f"{CLASS_ID} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

    except Exception as e:
        print(f"Error parsing filename {filename}: {e}. Skipping.")
        return None


def generate_data_yaml(output_dir, class_name, num_classes):
    """生成YOLOv8所需的data.yaml配置文件"""
    yaml_content = f"""
# Dataset root directory
path: {os.path.abspath(output_dir)}

# Train/val/test images directories
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {num_classes}

# Class names
names: ['{class_name}']
"""
    yaml_path = os.path.join(output_dir, 'ccpd2019.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nGenerated data.yaml at: {yaml_path}")


# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(CCPD_BASE_DIR):
        print(f"Error: CCPD_BASE_DIR '{CCPD_BASE_DIR}' does not exist.")
        print("Please modify CCPD_BASE_DIR to your actual CCPD2019 base dataset path.")
        exit()

    create_yolo_dataset_structure(OUTPUT_YOLO_DATASET_DIR)

    all_image_files = [f for f in os.listdir(CCPD_BASE_DIR) if f.lower().endswith('.jpg')]
    random.shuffle(all_image_files)  # 打乱图片顺序，以便随机划分数据集

    total_images = len(all_image_files)
    train_split_idx = int(total_images * TRAIN_RATIO)
    val_split_idx = int(total_images * (TRAIN_RATIO + VAL_RATIO))

    print(f"\nFound {total_images} images. Splitting into:")
    print(f"  Train: {train_split_idx} images")
    print(f"  Val: {val_split_idx - train_split_idx} images")
    print(f"  Test: {total_images - val_split_idx} images")
    print("\nProcessing images...")

    processed_count = 0
    for i, img_filename in enumerate(all_image_files):
        img_path = os.path.join(CCPD_BASE_DIR, img_filename)
        img_name_without_ext = os.path.splitext(img_filename)[0]

        # 确定当前图片属于哪个数据集子集
        subset = ''
        if i < train_split_idx:
            subset = 'train'
        elif i < val_split_idx:
            subset = 'val'
        else:
            subset = 'test'

        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            yolo_label_line = parse_ccpd_filename(img_filename, img_width, img_height)

            if yolo_label_line:
                # 写入标签文件
                label_output_path = os.path.join(OUTPUT_YOLO_DATASET_DIR, 'labels', subset,
                                                 f"{img_name_without_ext}.txt")
                with open(label_output_path, 'w') as f:
                    f.write(yolo_label_line)

                # 复制图片文件
                image_output_path = os.path.join(OUTPUT_YOLO_DATASET_DIR, 'images', subset, img_filename)
                shutil.copy(img_path, image_output_path)
                processed_count += 1
            else:
                print(f"Skipped {img_filename} due to parsing error or invalid bounding box.")

        except Exception as e:
            print(f"Error processing {img_filename}: {e}. Skipping.")

        if (i + 1) % 1000 == 0 or (i + 1) == total_images:
            print(f"Processed {i + 1}/{total_images} images...", end='\r')

    print(f"\n\nFinished processing. Successfully converted {processed_count} images.")

    # 生成data.yaml
    generate_data_yaml(OUTPUT_YOLO_DATASET_DIR, CLASS_NAME, 1)

    print("\nDataset preparation complete! You can now start training with YOLOv8.")
    print(f"Your dataset is located at: {os.path.abspath(OUTPUT_YOLO_DATASET_DIR)}")
    print(f"The data.yaml file is: {os.path.abspath(os.path.join(OUTPUT_YOLO_DATASET_DIR, 'ccpd2019.yaml'))}")