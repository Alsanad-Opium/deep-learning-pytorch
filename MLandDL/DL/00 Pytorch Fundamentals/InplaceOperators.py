#Inplace operators are the operators that directly modify the content of the tensor without making an additional memory allocation for it

#The inplace operator can be used after any operator by adding an underscore after the name of the methods eg m.add_(), m.relu_()

import torch 

m =  torch.rand(2,2)
n = torch.rand(2,2)

print(m,n)
print(m.data_ptr())# prints the memoery of the m tensor
print("Using the inplace operator sub_() n fromm m")
# Here the reuslt will be stored on the same memory as the m we can that by using the 

m.sub_(n)
print(m.data_ptr())#print the memoery address of tensor m 
print(m)

print("Using the relu_() inpalce operator on m which now has the result of the subtraction between m and n")
m.relu_()
print(m)

# the benefits of using the inplace oprators is that they save memoery allocations and can be faster in some cases 
new_tensor = torch.rand(1000,1000)
print(new_tensor.data_ptr())
new_tensor.add_(10)
print(new_tensor.data_ptr())
print(new_tensor)
