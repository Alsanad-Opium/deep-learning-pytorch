
# 📘 PyTorch Tensor Notes  

## 1. Checking PyTorch Version
```python
import torch
print(torch.__version__)
```
- Prints the installed PyTorch version.

---

## 2. Scalars (0D Tensor)
```python
scalar = torch.tensor(7)
print(scalar)          # tensor(7)
print(scalar.ndim)     # 0 → because no brackets
print(scalar.item())   # 7 → converts tensor to a normal Python int
```
- **Scalar** = A single number.  
- `.ndim` → number of dimensions (scalar = 0D).  
- `.item()` → extracts the value as a plain integer/float.  

---

## 3. Vectors (1D Tensor)
```python
vector = torch.tensor([2, 2, 23, 123])
print(vector)          # tensor([2, 2, 23, 123])
print(vector.ndim)     # 1 → one pair of brackets
print(vector.shape)    # torch.Size([4]) → 4 elements
```
- **Vector** = List of numbers in a line (1D tensor).  
- `.ndim` → counts dimensions.  
- `.shape` → shows structure (length of vector).  

💡 **Difference:**
- `ndim`: Number of dimensions (how many brackets).  
- `shape`: How many rows & columns in each dimension.  

---

## 4. Matrices (2D Tensor)
```python
matrix = torch.tensor([[4,5,6],
                       [1,2,3]])
print(matrix.ndim)     # 2 → two brackets
print(matrix.shape)    # torch.Size([2, 3]) → 2 rows, 3 cols
print(matrix[1])       # tensor([1,2,3]) → indexing
```
- **Matrix** = Numbers arranged in rows & columns (2D tensor).  
- Shape = (rows, columns).  

---

## 5. Random Tensors
```python
random = torch.rand(3,5)
print(random)
print(random.ndim)     # 2
print(random.size())   # torch.Size([3,5])
```
- **Why random tensors?**  
  - Neural networks start with random values → adjusted during training.  
- `torch.rand(shape)` → creates tensor with random values between 0 and 1.  

---

## 6. Range Tensor
```python
range_tensor = torch.arange(1,10)
print(range_tensor)    # tensor([1,2,3,4,5,6,7,8,9])
```
- `torch.arange(start, end)` → creates values from start to (end-1).  

---

# 📝 Quick Recap
- **Scalar (0D):** Single number → `torch.tensor(7)`  
- **Vector (1D):** Line of numbers → `torch.tensor([1,2,3])`  
- **Matrix (2D):** Rows & columns → `torch.tensor([[1,2],[3,4]])`  
- **Random Tensor:** `torch.rand(shape)`  
- **Range Tensor:** `torch.arange(start, end)`  
