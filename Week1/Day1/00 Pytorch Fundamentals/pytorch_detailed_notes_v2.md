# 📘 PyTorch Tensor Notes — Detailed (Short & Simple explanations + examples)

This document explains common PyTorch tensor operations and attributes in short, simple language with small examples and expected behavior. Copy the examples into a Python file or Jupyter cell and run them to see the results.

---

## 0. Check PyTorch Version
**What it does:** Shows which PyTorch version is installed.
```python
import torch
print(torch.__version__)  # e.g., "2.2.0"
```

---

## 1. Create a Tensor (`torch.tensor`)
**What it does:** Builds a tensor from Python lists or numbers.
```python
scalar = torch.tensor(7)            # 0D tensor (scalar)
vector = torch.tensor([1, 2, 3])    # 1D tensor (vector)
matrix = torch.tensor([[1,2],[3,4]])# 2D tensor (matrix)
```
**Why use:** This is the most direct way to make tensors from values you know.

---

## 2. Number of Dimensions (`.ndim`)
**What it does:** Returns how many dimensions the tensor has (0, 1, 2, ...).
```python
print(scalar.ndim)   # 0
print(vector.ndim)   # 1
print(matrix.ndim)   # 2
```
**Short:** Count of brackets/layers (0 = scalar, 1 = vector, 2 = matrix).

---

## 3. Shape & Size (`.shape`, `.size()`)
**What it does:** Tells the size along each dimension (rows, cols, ...).
```python
print(vector.shape)  # torch.Size([3])  -> 3 elements
print(matrix.shape)  # torch.Size([2, 2]) -> 2 rows, 2 cols
print(matrix.size()) # same as .shape
```
**Short:** Use to check dimensions before operations to avoid shape errors.

---

## 4. Get Python Number from Tensor (`.item()`)
**What it does:** Extracts a single value from a 0D tensor to a Python int/float.
```python
print(scalar.item())  # 7 (Python int)
```
**Short:** Use when you need a plain number (e.g., logging or printing).

---

## 5. Create Random Tensors (`torch.rand`)
**What it does:** Creates a tensor filled with random numbers from [0, 1).
```python
r = torch.rand(2,3)   # 2 rows, 3 cols, values between 0 and 1
print(r.shape)        # torch.Size([2, 3])
```
**Short:** Useful for initializing model weights or testing shapes.

---

## 6. Create Ranges (`torch.arange`)
**What it does:** Generates a sequence of numbers like Python `range()`.
```python
a = torch.arange(1, 6)   # tensor([1,2,3,4,5])
```
**Short:** Quick way to create sequential numbers in a tensor.

---

## 7. Zeros with a Given Shape (`torch.zeros_like`, `torch.zeros`)
**What it does:** Make tensors filled with zeros matching a shape or explicit sizes.
```python
x = torch.arange(1,6)
zeros = torch.zeros_like(x)   # same shape as x, all zeros
zeros2 = torch.zeros(2,3)     # explicit 2x3 zeros tensor
```
**Short:** Good for placeholders, masks, or initializing accumulators.

---

## 8. Datatype (`.dtype`, `.type()`)
**What it does:** Shows or converts a tensor's data type (float, int, etc.)
```python
t = torch.tensor([1.2, 3.4])
print(t.dtype)                # e.g., torch.float32
t16 = t.type(torch.float16)   # convert to float16
print(t16.dtype)              # torch.float16
```
**Short:** Use correct dtype for operations (some ops require floats, some require ints).

---

## 9. Device (`.device`, `.to()`)
**What it does:** Shows where the tensor lives (CPU or GPU) and moves tensors.
```python
t = torch.rand(2,2)
print(t.device)               # e.g., cpu
# Move to GPU (if available): t = t.to("cuda") 
# then print(t.device) -> cuda:0
```
**Short:** Move to GPU to speed up large computations; check device to avoid errors.

---

## 10. Gradient Tracking (`requires_grad`)
**What it does:** Tells PyTorch to track operations for automatic differentiation.
```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
print(w.requires_grad)  # True
```
**Short:** Turn on for parameters you want gradients for (models' weights).

---

## 11. Basic Arithmetic (+, -, *, /) & Broadcasting
**What it does:** Element-wise operations between tensors (or tensor and scalar).
```python
t = torch.tensor([1.0, 2.0, 3.0])
print(t + 10)        # tensor([11., 12., 13.])
print(t * 2)         # tensor([2., 4., 6.])
```
**Broadcasting:** if shapes differ, PyTorch may expand the smaller one to match when allowed.

**Short:** Use for simple element-wise math; watch shapes for broadcasting rules.

---

## 12. Addition using `torch.add`
**What it does:** Adds two tensors element-wise (same as `+`).
```python
a = torch.rand(2,3)
b = torch.rand(2,3)
print(torch.add(a, b))  # element-wise add
```
**Short:** Functional version of `+`, can accept an `alpha` scaling factor.

---

## 13. Element-wise vs Matrix Multiplication (`*` vs `matmul` / `@`)
**What it does:** `*` multiplies element-by-element; `matmul` or `@` does linear algebra matrix multiplication.
```python
A = torch.rand(2,3)
B = torch.rand(2,3)
print(A * B)                 # element-wise (shape 2x3)
# For matrix multiply, shapes must align: (m x n) @ (n x p) -> (m x p)
v = torch.tensor([1.,2.,3.]) # shape [3]
M = torch.rand(3,4)          # shape [3,4]
print(v @ M)                 # result shape [4]
print(torch.matmul(v, M))    # same as @
```
**Short:** Use `*` for per-element, `@` / `matmul` for dot products / matrix algebra.

---

## 14. Matrix Transpose (`.T`, `torch.transpose`)
**What it does:** Flips rows and columns (2D). `.T` is an easy shorthand.
```python
M = torch.tensor([[1,2,3],[4,5,6]])  # shape (2,3)
print(M.T.shape)                     # torch.Size([3,2])
# For higher dims use M.transpose(dim0, dim1)
```
**Short:** Needed when shapes don't align for matmul or to change axes order.

---

## 15. Aggregation (min, max, mean, sum, var, std)
**What it does:** Produces single-value summaries along the whole tensor or a dimension.
```python
t = torch.arange(1,11, dtype=torch.float32)  # 1..10

print(t.min())   # tensor(1.)
print(t.max())   # tensor(10.)
print(t.mean())  # tensor(5.5)
print(t.sum())   # tensor(55.)
print(t.var())   # variance
print(t.std())   # standard deviation
```
**With dimension:** add `dim=0` or `dim=1` to compute along an axis (returns tensor).
```python
M = torch.tensor([[1.,2.],[3.,4.]])
print(M.mean(dim=0))  # column-wise mean -> tensor([2., 3.])
print(M.mean(dim=1))  # row-wise mean -> tensor([1.5, 3.5])
```
**Short:** Use to reduce data for loss, metrics, or statistics.

---

## 16. Index of min / max (`argmin`, `argmax`)
**What it does:** Returns the index (position) of smallest/largest value.
```python
t = torch.tensor([5,1,9,3])
print(t.argmin())  # 1 (index of value 1)
print(t.argmax())  # 2 (index of value 9)
```
**Short:** Useful when you need the location of best/worst item (e.g., predicted class).

---

## 17. Reshape (`.reshape`) vs View (`.view`)
**What they do:** Change tensor shape.
- `.reshape(new_shape)`: returns a tensor with the new shape (may copy data).
- `.view(new_shape)`: returns a view that shares memory (faster but requires contiguous memory).
```python
t = torch.arange(1,10)       # shape [9]
r = t.reshape(3,3)           # shape [3,3]
v = t.view(3,3)              # also shape [3,3], shares memory with t
v[0,0] = 100                 # changes t too if view was used
```
**Short:** Use `.reshape` when unsure; use `.view` when you want a memory-sharing view (careful!).

---

## 18. Stack (`torch.stack`) and Concatenate (`torch.cat`, `vstack`, `hstack`)
**What they do:** Combine multiple tensors.
```python
t1 = torch.tensor([[1,2],[3,4]])
t2 = torch.tensor([[5,6],[7,8]])

S0 = torch.stack([t1, t2])         # new dim at front -> shape [2,2,2]
C = torch.cat([t1, t2], dim=0)     # join along rows -> shape [4,2]
V = torch.vstack([t1, t2])         # same as cat dim=0
H = torch.hstack([t1, t2])         # join side-by-side -> shape [2,4]
```
**Short:** `stack` adds a new axis, `cat` joins along an existing axis.

---

## 19. Squeeze & Unsqueeze (`.squeeze`, `.unsqueeze`)
**What they do:** Remove or add dimensions of size 1.
```python
t = torch.tensor([[1,2,3]])   # shape [1,3]
s = t.squeeze()               # shape [3]   (removed the 1-dim)
u = s.unsqueeze(0)            # shape [1,3] (adds a 1-dim back)
```
**Short:** Use to ensure tensors have expected number of dims (e.g., batch dim).

---

## 20. Permute (`.permute`) — reorder axes
**What it does:** Rearranges dimensions (useful for images or multi-dim data).
```python
x = torch.randn(2, 3, 4)   # shape [2,3,4]
y = x.permute(2, 0, 1)     # new shape [4,2,3]
```
**Short:** Reorder axes when a specific arrangement is required (e.g., channels-first vs channels-last).

---

## 21. Important Safety Checks & Tips
- **Shape mismatch** is the most common error. Always check `.shape` before matmul or broadcasting ops.
- **Datatype mismatch** can break aggregation functions (use `dtype=torch.float32` for mean/var/std).
- **Device mismatch** (CPU vs CUDA) causes runtime errors — move tensors to the same device with `.to(device)`.
- Use **small examples in REPL/Jupyter** to verify behavior before using large tensors in models.

---

## 22. Short Example: From creation → operation → stat
```python
import torch

# create
a = torch.arange(1,7, dtype=torch.float32).reshape(2,3)
b = torch.rand(2,3)

# op
c = a + b                    # element-wise add
d = c.mean(dim=1)            # mean along each row (returns shape [2])
min_index = c.argmin()       # index in flattened tensor
print(a, b, c, d, min_index)
```

---

## Quick Recap (one-line)
- `.ndim`: number of dimensions.  
- `.shape` / `.size()`: shape.  
- `.dtype`: datatype.  
- `.device`: CPU/GPU.  
- `.item()`: turn single-value tensor to Python scalar.  
- `torch.rand()`, `torch.arange()`, `torch.zeros_like()`: ways to create tensors.  
- `+`, `-`, `*`, `/`, `torch.matmul()` / `@`: arithmetic vs matrix multiply.  
- `reshape`, `view`, `stack`, `cat`, `squeeze`, `unsqueeze`, `permute`: change or combine shapes.  
- `min/max/mean/sum/var/std`, `argmin/argmax`: aggregate functions.  

---
