# 📘 PyTorch Notes — Indexing, NumPy Interop, Reproducibility & GPU (Generated)

These notes are generated from your script and explain each section in short, clear language with runnable examples and solved exercises. Save as `pytorch_indexing_repro_gpu_notes.md`.

---

## Table of Contents
1. Indexing in PyTorch (selecting data from tensors)  
2. PyTorch ↔ NumPy interoperability (shared memory caveats)  
3. Reproducibility (random seeds)  
4. GPU acceleration with PyTorch (checking devices, moving tensors)  
5. Exercises (questions from your script with full solutions)  
6. Quick Tips & Common Pitfalls  

---

## 1 — Indexing in PyTorch (selecting data from tensors)
**What it is:** Indexing lets you select parts (sub-tensors) of a tensor. You can use integer indices, slices, boolean masks, and fancy (list) indexing.

**Basic examples:**
```python
import torch
M = torch.arange(1,13).reshape(3,4)
# tensor([
#  [ 1,  2,  3,  4],
#  [ 5,  6,  7,  8],
#  [ 9, 10, 11, 12]
# ])

# Row 2 (0-based indexing)
row2 = M[1]        # tensor([5,6,7,8])

# Column 3
col3 = M[:, 2]     # tensor([3,7,11])

# Single element (row 1, col 2)
elem = M[1,2]      # tensor(7)

# Keep dims (slice to preserve batch dimension)
x = M[0:1, :]      # shape torch.Size([1,4]) - preserves 2D shape
```

**Fancy indexing (select arbitrary rows/cols):**
```python
selected_rows = M[[0,2], :]   # picks rows 0 and 2 -> shape (2,4)
```

**Boolean mask indexing:**
```python
mask = M % 2 == 0    # selects even numbers
evens = M[mask]      # flattened 1D tensor of even elements
```

**Important behaviors:**
- Fancy indexing returns a **copy** (modifying it won't change the original).
- Boolean masking returns a **1D** flattened result of matching elements.
- Use slicing (`:`) to preserve dimensions (useful when keeping batch axis).

---

## 2 — PyTorch & NumPy interoperability (shared memory details)
PyTorch uses NumPy heavily; conversions are straightforward but share memory by default.

**NumPy → PyTorch:**
```python
import numpy as np
arr = np.arange(1,10)        # numpy array
tensor = torch.from_numpy(arr)  # shares memory with arr
```
- After this, `arr` and `tensor` share the same memory. Changing one affects the other:
```python
arr[0] = 99
print(tensor[0])  # shows 99
```

**PyTorch → NumPy:**
```python
t = torch.ones(10)
arr_back = t.numpy()         # shares memory with t (CPU tensors only)
```
- Mutating `t` or `arr_back` will reflect in the other.

**Caveats & tips:**
- `.numpy()` works only for **CPU** tensors. If the tensor is on GPU, first move it to CPU: `tensor.cpu().numpy()`.
- If you want to avoid shared-memory side effects, create copies:
  - NumPy side: `arr_copy = arr.copy()`
  - PyTorch side: `tensor_copy = tensor.clone()`

**Dtype mapping:**
- NumPy `int64` → `torch.int64`
- NumPy `float64` → `torch.float64`
- PyTorch often uses `float32` for many ops; cast explicitly if needed: `tensor.float()`.

---

## 3 — Reproducibility (random seeds)
Neural networks rely on random initialization. To reproduce experiments, control random seeds.

**CPU RNG:**
```python
torch.manual_seed(42)
a = torch.rand(3,4)
```

**Comparing random draws:**
```python
torch.manual_seed(42)
r1 = torch.rand(3,4)
torch.manual_seed(42)
r2 = torch.rand(3,4)
# r1 == r2 (elementwise) -> True for every element
```

**Key point:** If you want identical random tensors across separate draws, set the seed immediately before each draw.

**GPU RNG:**
```python
torch.cuda.manual_seed(1234)           # seed for current CUDA device
torch.cuda.manual_seed_all(1234)       # seed all CUDA devices
```

**More deterministic behavior:**
- Use `torch.use_deterministic_algorithms(True)` to force deterministic algorithms (may raise errors for nondeterministic ops).
- For cuDNN:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Note: enabling strict determinism may slow down execution.

---

## 4 — GPU Acceleration with PyTorch (checking & moving devices)
**Check GPU availability:**
```python
torch.cuda.is_available()     # True if CUDA GPU exists and is usable
torch.cuda.device_count()     # number of GPUs
torch.cuda.current_device()   # current device index (0,1,...)
```

**Move tensors between devices:**
```python
t = torch.tensor([1,2,3])
print(t.device)            # cpu

t_gpu = t.to("cuda")       # move to GPU (if available)
print(t_gpu.device)        # cuda:0 (for example)

t_back = t_gpu.to("cpu")   # back to CPU
arr = t_back.numpy()       # now safe to convert to numpy
```

**Device mismatch error:** Operations between tensors on different devices will fail. Ensure all operands are on the same device.

**Non-blocking copies & pinned memory:**
- Use `pin_memory=True` in `DataLoader` for faster host→device transfer.
- Use `.to(device, non_blocking=True)` only if source is pinned, otherwise set `non_blocking=False`.

---

## 5 — Exercises (from your script) — questions + complete solutions

> These exercises are the same ones you used in your script. Solutions include code and brief explanations.

### Q1 — Create a random tensor with shape (7, 7).
**Solution**
```python
random_tensor = torch.rand(7,7)
print(random_tensor.shape)  # torch.Size([7,7])
```
**Explanation:** `torch.rand(7,7)` produces a 7×7 tensor of values in `[0,1)`.

---

### Q2 — Multiply Q1 tensor with another random tensor of shape (1, 7) (matrix multiplication).
**Solution**
```python
random_tensor2 = torch.rand(1,7)        # shape (1,7)
random_tensor2_T = random_tensor2.T     # shape (7,1)
result = random_tensor @ random_tensor2_T  # (7,7) @ (7,1) -> (7,1)
print(result.shape)  # torch.Size([7,1])
```
**Explanation:** Matrix multiplication requires inner dimensions match: (7,7) @ (7,1) → (7,1).

---

### Q3 — Set the random seed to 0 and repeat Q1 & Q2. Are results reproducible?
**Solution**
```python
torch.manual_seed(0)
r1 = torch.rand(7,7)

torch.manual_seed(0)
r2 = torch.rand(1,7).T

print(torch.equal(r1, torch.rand(7,7)))  # True when seed reset before draw
```
**Explanation:** Setting same seed before draws produces identical random tensors. Reset seed to reproduce draws.

---

### Q4 — Is there a GPU equivalent to `torch.manual_seed()`? Set GPU seed to 1234.
**Solution**
```python
torch.cuda.manual_seed(1234)
r_gpu1 = torch.rand(7,7, device="cuda")

torch.cuda.manual_seed(1234)
r_gpu2 = torch.rand(7,7, device="cuda")

print(torch.equal(r_gpu1, r_gpu2))  # True if seeded correctly
```
**Explanation:** `torch.cuda.manual_seed()` sets CUDA RNG for reproducible GPU draws. For multi-GPU, prefer `torch.cuda.manual_seed_all()`.

---

### Q5 — Create two random tensors shape (2,3) on GPU with seed 1234 (use `torch.manual_seed`).
**Solution**
```python
torch.manual_seed(1234)
a = torch.rand(2,3).to("cuda")
torch.manual_seed(1234)
b = torch.rand(2,3).to("cuda")
print(torch.equal(a,b))  # True
```
**Note:** `torch.manual_seed` affects CPU RNG used to create the tensors; moving to GPU keeps values same. For GPU-side RNG control, use `torch.cuda.manual_seed_all()`.

---

### Q6 — Matrix multiplication on tensors from Q5 (adjust shapes).
**Solution**
```python
b_T = b.T               # shape (3,2)
res = a @ b_T           # (2,3) @ (3,2) -> (2,2)
print(res.shape)        # torch.Size([2,2])
```
**Explanation:** Inner dims align (3), resulting in (2,2).

---

### Q7 — Find max and min values of result.
**Solution**
```python
mx = res.max().item()
mn = res.min().item()
print("Max:", mx, "Min:", mn)
```
**Explanation:** `.max()` and `.min()` return 0D tensors; `.item()` converts to Python scalars.

---

### Q8 — Find indices of max and min element (2D coordinates).
**Solution**
```python
idx_max = res.argmax().item()  # flattened index
idx_min = res.argmin().item()

# Convert flattened index to (row, col)
row_max, col_max = divmod(idx_max, res.shape[1])
row_min, col_min = divmod(idx_min, res.shape[1])
print("Max at:", (row_max, col_max))
print("Min at:", (row_min, col_min))
```
**Alternative (if available):**
```python
coords = torch.unravel_index(idx_max, res.shape)
```

---

### Q9 — Create random tensor shape (1,1,1,10) with seed=7 then squeeze to shape (10).
**Solution**
```python
torch.manual_seed(7)
t = torch.rand(1,1,1,10)
print("Original shape:", t.shape)  # (1,1,1,10)

s = t.squeeze()
print("Squeezed shape:", s.shape)  # (10,)
```
**Explanation:** `.squeeze()` removes axes of size 1.

---

## 6 — Quick Tips & Common Pitfalls (from this script)
- **Avoid shadowing builtins:** don't name variables `range`, `list`, `input`, etc. Use `range_tensor` or `rng`.
- **NumPy arrays have no `.device` attribute.** Query device on tensors only.
- **`.numpy()` only for CPU tensors.** Call `.cpu()` first for GPU tensors.
- **Seed semantics:** `torch.manual_seed()` controls CPU draws; `torch.cuda.manual_seed_all()` + `torch.cuda.manual_seed()` for GPU draws.
- **Comparisons:** `tensorA == tensorB` yields elementwise boolean tensor; use `torch.equal` for exact full equality; use `torch.allclose` for float comparisons.
- **Remember to set seeds before each random draw** if you want exact repetition in a sequence of draws.

---

### Saved file
This Markdown file was generated as `pytorch_indexing_repro_gpu_notes.md`.  
You can download it from the environment.

