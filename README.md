# LPR_YOLOv8_paddleOCR
Based on Yolov8,Data From CCPD2019
车牌数据集来自CCPD2019——中文国城市车牌开源数据集。下载链接于https://pan.baidu.com/share/init?surl=i5AOjAbtkwb17Zy-NQGqkw，密码为hm0u
由于原始数据集有24G之多，而对于YOLOv8训练以作为车牌定位而言过多，这里使用base文件夹中的随机挑选4G数据集，大约有4万张左右。
原始数据集的格式为jpg，标签即图片名。
![image](https://github.com/user-attachments/assets/97681199-400b-48c8-bdc0-1726be2b00e9)

transform.py将数据集转换为YOLO可以训练的格式，需要指定文件夹。
yolov8.py即训练代码，best.pt即经过80轮epoch后得到的最佳模型，可以直接使用
OCR.py使用的paddleOCR，CPU版本，若需要GPU版本需要CUDA与paddleOCR的版本一致
