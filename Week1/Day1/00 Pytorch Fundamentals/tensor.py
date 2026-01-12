import torch 


## this prints the version of the pytorch installed on your system
print(torch.__version__)


# this is how you create a tensor in python using pytorch library

scalar  = torch.tensor(7)

print(scalar)

# how to know how many dimensions a tensor can have
dim = scalar.ndim
print(dim)


# to remove the type tensor and jjst represent it as a normal integer we use this 
# item = scalar.item()

print(scalar.item())

#vector A vector is just a list of numbers arranged in a single line.

# Example: [1, 2, 3]
# This is called a 1D tensor in PyTorch/Deep Learning.

# So, you can think of a vector as:
# 👉 a line of numbers that represents something.

# 3. How (it works)?

# A vector has two key things:

# Direction

# Magnitude (size/length)

# Example: [3,4] → This vector points in the direction of x=3, y=4.
# Its magnitude = √(3² + 4²) = 5.


vector = torch.tensor([2,2,23,123])
print(vector)

print("Using ndim")
print(vector.ndim)
print("Using the shape")
print(vector.shape)
# here is the differnce between ndim and the shape method 

# ndim = IT return the number of pair of square brackets 
# where as 
# shape = returns the number of the rows and the columns 

matrix = torch.tensor([[4,5,6],
                       [1,2,3]])

print("ON MATRIX using the ndim method ")
print(matrix.ndim)

print("using the shape on MATRIX")

print(matrix.shape)

# the index starts from 0 to 1,2,3,....

print(matrix[1])


# anything above this becomes the tensor 



### RANDOM TENSORS  (VERY IMPORTANT)

# WHY RANDOM TENSORS?

# Random tensors are important because the way most of the neural nwtworks works is through the random number at first and then it 
# then is adjust those random numbers to better represent the data.

# tHIS IS HOW YOU CREATE RANDOM TENSOR 
# the rand use the inputs same as the shape and the size 
random  = torch.rand(3,5)

print(random)

print(random.ndim)

print(random.size())


# we can create range of  tensor like this 

range  = torch.arange(1,10)

print(range)

# we can also create tensors of all zeros or all ones for the above shape by using this  method

zeroes = torch.zeros_like(input= range) # the input is the tensor whose shape we want to copy 
print(zeroes)


# Tensor Datatype (Tensor Attributes)

# Note:- tensor  datatypes is one of the 3 big errors that you will run into with pytroch  and deep learning
# Tensors not right datatype 
# tensors not right shape 
# tensor not on right device (cpu , gpu)

float_32_tensor  =  torch.tensor ([1.23,43.34,52], dtype=None, #what datatype is the tensor (eg float32, float 16 , int )
                                  device=None, # default is the cpu but you can change it to gpu cuda in future 
                                  requires_grad=False # whether or not to track gradients with tensors opereations this tensor 
                                  )

print(float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)
print(float_16_tensor.dtype)
 
#refer the torch.tensor datatype doc for more tensor dtypes
 # the above steps can be done to tackle the tensor not right datatpye issue


 # Getting information from tensor 

 # Tensors not right datatype :-  we can use the tensor.dtype
# tensors not right shape :- to get the shape we can us eth tensor.shape or size
# tensor not on right device (cpu , gpu):- to get the device from the tensor we can use the tensor.device method 


some_tensor = torch.rand(3,4)

print(some_tensor)

print("Datatype of a tensor ",some_tensor.dtype)
new_some_tensor_16_float = some_tensor.type(torch.float16)
print("New tensors dtype ",new_some_tensor_16_float.dtype)
print("Shape of a tensor ",some_tensor.shape)
print("Device on which the tensor is created ",some_tensor.device)


## Mnipulating tensors (tensor Operations)

 ## various operation can be done on tensors 
 # addition 
 # Subtraction
  # Multiplication
 # division
 # matrix mulplication 

 # creatin a tensor

tencor = torch.tensor([1,2,3]) # this created a int64 tensor by default
tencor  = tencor.type(torch.float32)

print(tencor +10) # ADDS 10 TO EACH element of the tensor


print(tencor - 10)
print(tencor * 10)
print(tencor / 10)

tensor2 = torch.rand(3,4) 
tensor1 = torch.rand(3,4)


print(tencor @ tensor2) # this is the same as the above matmul function

print(torch.add(tensor1, tensor2))


# two ways of performing multiplication are

# element wise multiplication
print("Printing element wise multiplication")

print(tensor1 * tensor2)
# matrix mulplication (dot product)
print("Printing Matrix multiplication")

print(torch.matmul(tencor, tensor2)) 

# one of the most common wrror in deep learning is shape errors
# to avoid this we can use the shape method to check the shape of the tensor

# we can then cahnge the shaoe of the tensor by using the tensor.T method
# the tensor.T method return acopy of the tensor and does not change the original tensor


# Tensor Aggregation (min,max,mean,sum,etc)

tensor3 = torch.arange(1,11 , dtype=torch.float32) # here i entered the dtype beacuse some aggregation methods cannot work with int64 dtype  it will show the error got Long instead.
# the aggregation works with float32 dtype that why i have specified the dtype here as float32

print(tensor3)

print(tensor3.min())
print(tensor3.mean()) #cannot work with Long dtype
print(tensor3.max())
print(tensor3.sum())
print(tensor3.var())
print(tensor3.std())

# to get the index of the min and the 
# max value we can use the argmin nad th argmax function

print(tensor3)

print("Iusing the argmin and the argmax to get the index of the min and the max")

print(tensor3.argmin())
print(tensor3.argmax())

# reshaping, Stacking , sqeezing and unsqeezing tensors


# Reshaping - Reshapes the input tensor ti aa defined Shape
# View - returns a view of the tensor as a different shape but keeps th same memory as the original tensor
#Stacking -Combines multuple Tensors on top of each other (vstack, hstack) vstack - on top of each other. hstack() - side by side 
# squeezing - removes all the 1 dimension from the tensor
# un squeezing - adds a 1 dimension to the target tensor

#permute - returns a view of the input with dimensions permuted (swapped) in a certain way

tensor = torch.arange(1,10)

print(tensor)
# using the reahspe to change the shape of the tensor

reshaped_tensor = tensor.reshape(9,1)
print(reshaped_tensor)
print(reshaped_tensor.shape)


print("Using the view method noe what view does is so simple it jsut create acopy of the element or tensor and the new element shares the same memory as the original tensor memory")

view_tensor = tensor.view(3,3)
print(view_tensor)
# any changes done to the view of a tensor affects the original tensor

view_tensor[0,0] = 100
print(tensor)


tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Stack them along a new dimension (default dim=0)
stacked_tensor = torch.stack([tensor1, tensor2])
print(stacked_tensor)
# Output:
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])
# Shape: torch.Size([2, 2, 2])

# Stack them along a different dimension (e.g., dim=1)
stacked_tensor_dim1 = torch.stack([tensor1, tensor2], dim=1)
print(stacked_tensor_dim1)
# Output:
# tensor([[[1, 5],
#          [3, 7]],
#         [[2, 6],
#          [4, 8]]])
# Shape: torch.Size([2, 2, 2])

# stack them in differnt diemension with dim = 2

stacked_tensor_dim2 = torch.stack([tensor1,tensor2], dim = 2)
print(stacked_tensor_dim2)

# output of this as follows
# tensor([[[1, 5],
#          [2, 6]],

#         [[3, 7],
#          [4, 8]]])

# now Squeezing and unsqueezing of tensor

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

# Permute funciton

tensor6 = torch.randn(2,3,5,6)

print("The original dim of the tensor before permuting is (0,1,2,3)")
print(tensor6)
print(tensor6.shape)

print("The dimension will now be cahnged to (3,2,1,0)")
print(tensor6.permute(3,2,1,0))
print(f"The Permuted tensor shape :- {tensor6.shape}")


# Note - permute returns a view of the original tensor so any changes done to the permuted tensor will affect the original tensor


# Indexing In pytorch (selcting data from tensor)

#pytorch tensors & Numpy 

# numpy u s apopular scientifuc python numerical computing library 
# and beacasue of this pytorch uses the numopy for its installation 
# data in numpy want in pytorch tensor ->.from_numpy(ndarray)
#pytorch tensor -> numpy -> torch.tensor.numpy()

import numpy as np

array = np.arange(1,10)

tensor = torch.from_numpy(array)

print(array, tensor)

print(array.dtype, tensor.dtype)

array = array +1

print(array, tensor)

tensor  = torch.ones(10)

array = tensor.numpy()
print(tensor, "\n", array )
print(array.dtype, tensor.dtype)

# also remember to keep in mind the dtypes when converting from numpy tp tensor  or vice versa


# Repriducibilty (trying to take ranodm ou tof the random)

#in short how a neural network 

#starts with random number-. tensor operations ->update random numbers to tyr to make them some reperesentaion fo the data ->again ->again and again -> eventually the random numbers represent the data well

# tO REDUCE THE RANDOMNESS IN NEURAL NETWORKSS and pytrorch comes the concept of random seed 
# Essentially what the random seed does is flaourrs the randomness

random_tensorA = torch.rand(3,4)
random_tensorB = torch.rand(3,4)

print(random_tensorA)
print(random_tensorB)

print(random_tensorA == random_tensorB)

# now setting the random seed 

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

rand_tensor_C = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
rand_tensor_D = torch.rand(3,4)

print(rand_tensor_C)
print("********************************")
print(rand_tensor_D)

print(rand_tensor_C == rand_tensor_D)

# you have to set the manual seed for each time you need to create a tensor as that therandom is used once the seed is set and a tensor is created so remeber to create it everytime yo want to create a ranodm tensor 

# GPU Acceleration with pytorch 

print(torch.cuda.is_available())
print(torch.cuda.device_count())
# checking the current device whether cpu or gpu
print(torch.cuda.current_device())


tensor =  torch.tensor([1,2,3,4])

print(tensor, tensor.device)
# convertin the tensor from cpu to gpu
tensor = tensor.to("cuda")
print(tensor, tensor.device)


# Converting back to the cpu as numpy works only on cpu

tensor  = tensor.to("cpu")
array = tensor.numpy()

print(array, array.device)


#Exercises of what we ahve Learnt so far 

# question 1  Create a random tensor with shape (7, 7).

random_tensor = torch.rand(7,7)

print(random_tensor)
print(random_tensor.shape)

# Question 2:- Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).

random_tensor2 = torch.rand(1,7)

print(random_tensor2)

print("Changng the shape fo the tensor b for multiplication ")

random_tensor2 = random_tensor2.T

print(random_tensor2)

print(random_tensor @ random_tensor2)

# question 3 :- Set the random seed to 0 and do exercises 2 & 3 over again.


torch.manual_seed(0)

print("After setting the random seed to 0")
random1 = torch.rand(7,7)
print(random1)
torch.manual_seed(0)
random2 = torch.rand(1,7).T
print(random2)

print(random1 @ random2)

# Question 4 :- Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one). If there is, set the GPU random seed to 1234.

torch.cuda.manual_seed(1234)

random_tensor_onGpu = torch.rand(7,7, device="cuda")
print(random_tensor_onGpu)
torch.cuda.manual_seed(1234)
random_tensor_onGpu2 = torch.rand(7,7, device = "cuda")
print(random_tensor_onGpu2)

print(random_tensor_onGpu == random_tensor_onGpu2)


#question 5 :- Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).

torch.manual_seed(1234)
random_tensor = torch.rand(2,3).to("cuda")
torch.manual_seed(1234)
random_tensor1 = torch.rand(2,3).to("cuda")

print(random_tensor)
print(random_tensor1)


# Question 6 :- Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).

random_tensor1 = random_tensor1.T
print(random_tensor @ random_tensor1)

# Question 7 :-  Find the maximum and minimum values of the output of 7.
print("Maximum ")
print((random_tensor @ random_tensor1).max())
print("Minimum")
print((random_tensor @ random_tensor1).max())

#Question 8 :- Find the maximum and minimum index values of the output of 7
print("Maximum Element Index")
print((random_tensor @ random_tensor1).argmax())
print("Minimum Element Index")
print((random_tensor @ random_tensor1).argmin())


# Question 9 :- Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.

torch.manual_seed(7)

random_tensor4 = torch.rand(1,1,1,10)
print(random_tensor4)
print(random_tensor4.shape)

new_tensor = random_tensor4.squeeze()

print(new_tensor)
print(new_tensor.shape)