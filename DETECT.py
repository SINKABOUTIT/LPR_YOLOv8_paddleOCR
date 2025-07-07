import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version PyTorch was built with:", torch.version.cuda)
    print("cuDNN Version PyTorch was built with:", torch.backends.cudnn.version())
    print("Current CUDA Device:", torch.cuda.get_device_name(0))
else:
    print("PyTorch 未检测到 CUDA GPU。")

exit()