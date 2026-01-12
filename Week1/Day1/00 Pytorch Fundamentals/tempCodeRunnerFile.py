
import torch
tensor5 = torch.tensor([[1,2,3,4,5,6,7,8,9]])

print("Printing the tensor before and after squeezing is applied to them ", tensor5)

squeezed_tensor =tensor5.squeeze()
print(squeezed_tensor)

print(f"Shape of the tensor after squeezing {squeezed_tensor.shape}")

#unsqueezing of tensor
unsqueezeed_tensor = squeezed_tensor.unsqueeze(0)
# print(f"the tensor after unsqueezing {squeezed_tensor.unsqueeze(0)}")

print(unsqueezeed_tensor)

print(f"Shaoe of the tensor after the unsqeezing of the tensor {unsqueezeed_tensor.shape}")