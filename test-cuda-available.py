import torch
print(torch.__version__)          # check PyTorch version
print(torch.cuda.is_available())  # should return True
print(torch.cuda.get_device_name(0))  # should show GTX 1650
