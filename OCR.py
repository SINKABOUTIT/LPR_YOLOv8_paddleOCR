import os
import sys
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import logging



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    # 尝试导入所有必要的库
    from ultralytics import YOLO
    import torch
    from paddleocr import PaddleOCR
    # logging 库已经移到文件顶部导入

except ImportError as e:
    # 如果缺少任何库，弹出错误信息并退出程序
    messagebox.showerror("库未安装",
                         f"缺少必要的库：{e}\n请确保所有依赖都已安装。\n"
                         f"对于NVIDIA GPU用户，请运行: pip install ultralytics opencv-python matplotlib paddlepaddle-gpu paddleocr\n"
                         f"对于CPU用户，请运行: pip install ultralytics opencv-python matplotlib paddlepaddle paddleocr\n"
                         f"提示: 对于PaddleOCR，某些版本不再需要use_gpu参数。请查阅最新文档。"
                         f"当前强制使用CPU模式。")
    sys.exit(1)

# ====================================================================
# --- 全局配置参数 ---
# ====================================================================

# YOLO 模型文件路径
# 请确保此路径指向您训练好的YOLO模型（通常是best.pt文件）
GLOBAL_TRAINED_YOLO_MODEL_PATH = r'C:\Users\24631\PycharmProjects\pythonProject\.venv\IMAGE\CCPD2019\runs\detect\ccpd2019_yolov8s_det5\weights\best.pt'

# YOLO 检测置信度阈值：低于此值的检测框将被过滤
CONFIDENCE_THRESHOLD = 0.25
# YOLO IOU（交并比）阈值：用于非极大值抑制，过滤重叠的检测框
IOU_THRESHOLD = 0.7
# YOLO 模型输入图像尺寸
YOLO_IMG_SIZE = 640

# --- 强制使用 CPU 进行推理 ---
# YOLO 模型推理设备设置为 'cpu'。
# PaddlePaddle 会根据安装的paddlepaddle包（GPU或CPU版本）自动选择设备。
DEVICE = 'cpu'
print("已强制设置为 CPU 进行推理。速度可能较慢。")

# --- PaddleOCR 初始化配置 ---
# 设置 PaddleOCR 日志级别，只显示警告和错误，减少控制台输出噪音
logging.getLogger('ppocr').setLevel(logging.WARNING)

# 初始化 PaddleOCR 识别器
# lang='ch' 表示加载中文识别模型。
# 注意：新版本PaddleOCR通常不需要use_gpu、use_textline_orientation等参数在构造函数中显式指定，
# 它会根据环境（paddlepaddle或paddlepaddle-gpu）和默认配置自动处理。
print("DEBUG: 正在初始化 PaddleOCR。首次运行可能需要下载模型...")
try:
    ocr = PaddleOCR(lang='ch')
    print("DEBUG: PaddleOCR 初始化完成。")
except Exception as e:
    messagebox.showerror("PaddleOCR 初始化错误",
                         f"PaddleOCR 初始化失败: {e}\n"
                         f"请检查网络连接以下载模型，或确认您的PaddlePaddle和PaddleOCR版本兼容。")
    sys.exit(1)



def run_lpr_system(yolo_model_path, conf_thresh, iou_thresh, yolo_imgsz, device):
    """
    运行车牌检测和识别系统。
    加载YOLO模型，通过文件选择图片，然后对图片进行检测和识别。
    """
    print("DEBUG: 进入 run_lpr_system 函数。")

    # --- 文件选择对话框 ---
    root = tk.Tk()
    root.withdraw()  # 隐藏Tkinter主窗口，只显示文件对话框
    print("DEBUG: Tkinter 根窗口创建并隐藏。")

    file_path = None  # 初始化文件路径变量
    try:
        # 弹出文件选择对话框，等待用户选择图片
        file_path = filedialog.askopenfilename(
            title="选择要测试的图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        print(f"DEBUG: filedialog.askopenfilename 返回: {file_path}")
    except Exception as e:
        # 捕获文件对话框可能发生的错误（例如，X11 DISPLAY问题）
        messagebox.showerror("Tkinter 错误", f"打开文件选择对话框失败: {e}\n请检查您的Tkinter环境或显示设置。")
        root.destroy()  # 尝试销毁Tkinter实例
        return

    # 如果用户没有选择文件，则退出程序
    if not file_path:
        messagebox.showinfo("取消", "未选择图片，程序退出。")
        root.destroy()  # 销毁Tkinter实例，确保程序完全退出
        return

    print("DEBUG: 已选择图片，路径为:", file_path)
    test_image_path = file_path

    # --- YOLO 模型加载 ---
    # 检查YOLO模型文件是否存在
    if not os.path.exists(yolo_model_path):
        messagebox.showerror("错误", f"未找到YOLO模型文件在 {yolo_model_path}\n请确认路径正确。")
        root.destroy()
        return

    print(f"正在加载YOLO模型: {yolo_model_path} 到 {device}")
    yolo_model = YOLO(yolo_model_path)  # YOLO模型加载时会根据device参数自动处理
    print("DEBUG: YOLO 模型加载完成。")

    # --- 图片读取 ---
    print(f"正在读取图片: {test_image_path}")
    original_image = cv2.imread(test_image_path)
    if original_image is None:
        messagebox.showerror("错误", "无法读取图片，请检查文件是否损坏或路径是否正确。")
        root.destroy()
        return
    print("DEBUG: 图片读取成功。")

    # --- YOLO 推理 (车牌检测) ---
    print(f"正在对图片进行YOLO推理...")
    # YOLO推理，device参数会控制它在CPU上运行
    yolo_results = yolo_model.predict(
        source=test_image_path,
        conf=conf_thresh,
        iou=iou_thresh,
        save=False,  # 不保存检测结果图片到磁盘
        save_txt=False,  # 不保存检测框坐标到txt
        save_conf=False,  # 不保存置信度到txt
        show=False,  # 不显示实时推理窗口
        imgsz=yolo_imgsz,  # 模型输入图像尺寸
        device=device  # 推理设备 (CPU)
    )
    print("DEBUG: YOLO 推理完成。")

    print("\n--- YOLO 检测完成，开始OCR识别 ---")

    plate_detection_count = 0
    recognition_summary_str = ""
    annotated_image_rgb = None  # 初始化，以防没有检测结果

    # --- 处理YOLO检测结果，进行OCR识别 ---
    for result_obj in yolo_results:
        # 获取YOLO绘制后的图像，用于最终显示
        annotated_image = result_obj.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        if len(result_obj.boxes) > 0:  # 如果检测到车牌
            plate_detection_count = len(result_obj.boxes)
            recognition_summary_str += f"在图片 '{os.path.basename(test_image_path)}' 中检测到以下车牌：\n"
            print(f"DEBUG: 检测到 {plate_detection_count} 个车牌。")

            for i, box in enumerate(result_obj.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框的像素坐标
                yolo_conf = box.conf[0].item()  # 获取YOLO检测置信度

                img_h, img_w, _ = original_image.shape
                # 确保裁剪边界在图像范围内，防止越界错误
                x1_clip = max(0, x1)
                y1_clip = max(0, y1)
                x2_clip = min(img_w, x2)
                y2_clip = min(img_h, y2)

                cropped_plate_img = original_image[y1_clip:y2_clip, x1_clip:x2_clip]
                print(
                    f"DEBUG: 裁剪车牌 {i + 1}，原始坐标 ({x1}, {y1}) 到 ({x2}, {y2})，裁剪尺寸: {cropped_plate_img.shape}")

                # 检查裁剪后的图像是否有效（非空）
                if cropped_plate_img.shape[0] == 0 or cropped_plate_img.shape[1] == 0:
                    print(f"警告: 裁剪车牌 {i + 1} 尺寸为0，跳过OCR。")
                    recognition_summary_str += f"  - 车牌 {i + 1}: (YOLO置信度: {yolo_conf:.2f}) 裁剪失败，无法识别。\n"
                    continue

                # 使用PaddleOCR进行识别。对于裁剪好的单行图片，ocr.ocr() 会自动进行文本行识别模式。
                ocr_output = ocr.ocr(cropped_plate_img)
                print(f"DEBUG: OCR对车牌 {i + 1} 的输出: {ocr_output}")

                recognized_text = "未识别到"
                ocr_confidence = 0.0

                # --- 修正后的OCR结果解析逻辑 ---
                # 根据你之前提供的输出：DEBUG: OCR对车牌 1 的输出: [[[[13.0, 3.0], [253.0, 47.0], ...], ('赣E·46719', 0.9690245389938354)]]
                # 这表明ocr_output是一个列表，其第一个元素（ocr_output[0]）是一个包含两个元素的列表/元组：
                # 第一个元素是检测框坐标列表，第二个元素是 (文本, 置信度) 元组。
                if ocr_output and len(ocr_output) > 0 and ocr_output[0] is not None:
                    # 确保 ocr_output[0] 存在且不是空列表/None
                    # 检查 ocr_output[0] 的第二个元素是否存在且是元组 (text, score)
                    first_item_in_output = ocr_output[0]  # 获取第一个识别结果项
                    if isinstance(first_item_in_output, list) and len(first_item_in_output) >= 2 and \
                            isinstance(first_item_in_output[1], tuple) and len(first_item_in_output[1]) >= 2:
                        recognized_text = first_item_in_output[1][0]  # 获取识别到的文本
                        ocr_confidence = first_item_in_output[1][1]  # 获取识别置信度
                    else:
                        print(f"DEBUG: OCR结果格式不符合预期，跳过解析。原始输出: {ocr_output}")

                recognition_summary_str += (
                    f"  - 车牌 {i + 1}: 坐标 ({x1}, {y1}) 到 ({x2}, {y2}), YOLO置信度: {yolo_conf:.2f}\n"
                    f"    OCR识别结果: {recognized_text} (OCR置信度: {ocr_confidence:.2f})\n"
                )

        else:  # 未检测到任何车牌
            recognition_summary_str = f"在图片 '{os.path.basename(test_image_path)}' 中未检测到车牌。"
            print("DEBUG: 未检测到任何车牌。")
            # 如果没有检测到，annotated_image_rgb 可能还是None，为了避免错误，给它一个空白图像
            if annotated_image_rgb is None:
                annotated_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # --- 显示结果 ---
    print("\n--- 最终识别摘要 ---")
    print(recognition_summary_str)
    messagebox.showinfo("检测与识别结果", recognition_summary_str)

    # 显示标注后的图片
    plt.figure(figsize=(12, 10))
    plt.imshow(annotated_image_rgb)
    plt.title(f"Detection and Recognition Results for {os.path.basename(test_image_path)}")
    plt.axis('off')  # 关闭坐标轴
    plt.show()

    # 完成所有操作后，销毁Tkinter实例，确保程序完全退出
    root.destroy()
    print("DEBUG: 程序执行完毕，Tkinter实例已销毁。")



if __name__ == '__main__':
    run_lpr_system(
        GLOBAL_TRAINED_YOLO_MODEL_PATH,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        YOLO_IMG_SIZE,
        DEVICE  # 传递 'cpu' 给 YOLO
    )