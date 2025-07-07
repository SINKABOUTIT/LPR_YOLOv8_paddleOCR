import os
from ultralytics import YOLO


DATA_YAML_PATH = 'Your data location'

# 可选: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
MODEL_WEIGHTS = 'yolov8s.pt' # 推荐从s开始，性能和速度有较好平衡

# 训练参数
TRAIN_EPOCHS = 80       # 训练轮数
IMAGE_SIZE = 640         # 输入图像尺寸 (例如 640, 960, 1280)
BATCH_SIZE = 32          # 批处理大小 (根据你的GPU显存调整)
PROJECT_NAME = 'runs/detect' # 训练结果将保存到这个目录下
EXPERIMENT_NAME = 'ccpd2019_yolov8s_det' # 本次训练的名称，作为子文件夹名

# 优化和高级参数
LEARNING_RATE = 0.01       # 初始学习率 (默认值，通常不需要改)
PATIENCE = 50              # 早停的耐心值，连续多少个epoch验证集mAP没有提升就停止 (默认50)
DEVICE = 0                 # 训练设备: 0表示第一个GPU, 'cpu'表示CPU (如果你有多个GPU，可以指定如1,2)

# 数据增强参数 (可以调整以提高精度)
FLIP_UD = 0.0              # 垂直翻转概率 (0.0 表示禁用，推荐禁用，车牌通常不会垂直翻转)
MOSAIC_PROB = 1.0          # Mosaic数据增强的概率 (默认1.0，如果小目标过多可能调低至0.5)

# --- 训练过程 ---

if __name__ == '__main__':


    model = YOLO(MODEL_WEIGHTS)

    print(f"Starting YOLOv8 training with model: {MODEL_WEIGHTS}")
    print(f"Dataset path: {DATA_YAML_PATH}")
    print(f"Output directory: {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}")

    # 开始训练
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=TRAIN_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        device=DEVICE,
        flipud=FLIP_UD,
        mosaic=MOSAIC_PROB,
        # 其他可能的参数，例如:
        # fliplr=0.5, # 水平翻转 (默认0.5，通常保留)
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # HSV增强 (默认值)
        # degrees=0.0, # 随机旋转角度
        # translate=0.1, # 随机平移
        # scale=0.5, # 随机缩放
        # perspective=0.0, # 透视变换
        # seed=42, # 设置随机种子以复现结果
        # dropout=0.0, # Dropout概率
        # plots=True, # 是否保存训练结果图表 (默认True)
        # val=True, # 是否在训练过程中进行验证 (默认True)
    )

    print("\nTraining complete!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Last model saved at: {model.trainer.last}")