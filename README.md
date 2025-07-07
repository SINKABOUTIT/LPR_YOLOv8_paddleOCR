# LPR_YOLOv8_paddleOCR
Based on Yolov8,Data From CCPD2019
车牌数据集来自CCPD2019——中文国城市车牌开源数据集。下载链接于https://pan.baidu.com/share/init?surl=i5AOjAbtkwb17Zy-NQGqkw，密码为hm0u
由于原始数据集有24G之多，而对于YOLOv8训练以作为车牌定位而言过多，这里使用base文件夹中的随机挑选4G数据集，大约有4万张左右。
原始数据集的格式为jpg，标签即图片名，格式如下
类别	描述	图片数
CCPD-Base	通用车牌图片	200k
CCPD-FN	车牌离摄像头拍摄位置相对较近或较远	20k
CCPD-DB	车牌区域亮度较亮、较暗或者不均匀	20k
CCPD-Rotate	车牌水平倾斜20到50度，竖直倾斜-10到10度	10k
CCPD-Tilt	车牌水平倾斜15到45度，竖直倾斜15到45度	10k
CCPD-Weather	车牌在雨雪雾天气拍摄得到	10k
CCPD-Challenge	在车牌检测识别任务中较有挑战性的图片	10k
CCPD-Blur	由于摄像机镜头抖动导致的模糊车牌图片	5k
CCPD-NP	没有安装车牌的新车图片	5k

transform.py将数据集转换为YOLO可以训练的格式，需要指定文件夹。
yolov8.py即训练代码，best.pt即经过80轮epoch后得到的最佳模型，可以直接使用
OCR.py使用的paddleOCR，CPU版本，若需要GPU版本需要CUDA与paddleOCR的版本一致
