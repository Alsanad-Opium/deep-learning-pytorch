# ✅ Advanced PyTorch Topics — Detailed Notes

This document covers the **advanced and missing topics** we hadn't fully covered before. Each section includes a short, clear explanation, practical tips, and runnable examples.

---

## 1) Broadcasting — exact rules + examples
**What it is:** letting PyTorch automatically expand smaller tensor shapes so you can do element-wise operations without manually reshaping.

**Rules (strict):**
- Compare shapes from **right to left**.
- Two dimensions are compatible if they are equal **or** one of them is `1`.
- Missing leading dimensions are treated as `1`.

**Examples**
```python
import torch

# Example A: scalar + tensor
t = torch.tensor([1., 2., 3.])      # shape (3,)
print((t + 10).shape)               # (3,)  -> scalar broadcast

# Example B: (3,1) broadcast to (3,4)
a = torch.rand(3,1)                 # shape (3,1)
b = torch.rand(3,4)                 # shape (3,4)
print((a + b).shape)                # (3,4)   because 1 -> 4 along last dim

# Example C: trailing mismatch fails
x = torch.rand(2,3)
y = torch.rand(3,2)
# x + y  -> Error: shapes (2,3) and (3,2) incompatible
```

**Pitfalls & tips**
- Broadcasting can hide bugs. If you expect a pairwise operation, ensure shapes match exactly.
- Use `.expand()` or `.repeat()` when you intentionally want to expand a dimension (`.expand()` returns a view).

---

## 2) Advanced Indexing (slices, boolean masks, fancy indexing)
**What it does:** extract sub-tensors in many ways.

**Examples**
```python
import torch
M = torch.arange(1,13).reshape(3,4)
# M =
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

# Basic slice:
print(M[1, :])        # second row -> tensor([5,6,7,8])

# Column selection:
print(M[:, 2])        # third column -> shape (3,)

# Fancy indexing (select rows 0 and 2):
print(M[[0,2], :])    # shape (2,4)

# Boolean mask:
mask = M % 2 == 0     # even numbers
print(M[mask])        # 1D tensor of even elements

# Preserve dims using slice:
x = M[0:1, :]         # shape (1,4)  (useful to preserve batch dim)
```

**Important behavior**
- Fancy indexing returns a **copy**, so modifying result won’t change the original tensor.
- Boolean mask indexing returns a **flattened** result.
- Use slices to preserve dimensions.

---

## 3) Memory & Views — `view()`, `reshape()`, `contiguous()`, `clone()`, NumPy sharing
**Core idea:** Views share memory; copies are independent.

**Details**
- `.view()` returns a view that **shares memory** (fast) but requires contiguous memory.
- `.reshape()` will return a view when possible, otherwise returns a copy.
- `.clone()` makes an independent copy.
- `.contiguous()` ensures the tensor has contiguous memory layout (useful before `.view()`).

**Examples**
```python
import torch
t = torch.arange(6)            # shape (6,)
v = t.view(2,3)                # view -> shares memory
v[0,0] = 99
print(t)                       # will reflect the change

# Force independent copy
v_copy = v.clone()
v_copy[0,0] = -1
print(t)                       # unchanged

# Non-contiguous example (transpose)
x = torch.arange(6).reshape(2,3).transpose(0,1)  # often non-contiguous
# x.view(...) may fail
xc = x.contiguous().view(3,2)  # make contiguous then view
```

### NumPy ↔ PyTorch shared memory
- `torch.from_numpy(np_arr)` shares memory with `np_arr`.
- `tensor.numpy()` shares memory with CPU tensor.
- Mutating one affects the other — use `.clone()` or `.copy()` to avoid this.

```python
import numpy as np
arr = np.arange(5)
t = torch.from_numpy(arr)
arr[0] = 99
print(t)  # shows the change
```

---

## 4) Comparisons: `==`, `torch.equal`, `torch.allclose`
**Behavior**
- `a == b` -> elementwise Boolean tensor.
- `torch.equal(a, b)` -> single boolean if shapes and all elements exactly equal.
- `torch.allclose(a, b)` -> True when values are close within tolerance (good for floating point).

**Examples**
```python
import torch
a = torch.tensor([1., 2., 3.])
b = torch.tensor([1., 2.0000001, 3.])
print(a == b)                 # elementwise -> tensor([True, False, True])
print(torch.equal(a, b))      # False (exact compare)
print(torch.allclose(a, b))   # True (within tolerance)
```

**Tip:** Use `allclose` for float comparisons in tests and assertions.

---

## 5) dtype mapping & pitfalls when converting from NumPy
**Rules**
- NumPy `int64` -> `torch.int64`
- NumPy `float64` -> `torch.float64`
- PyTorch ops often default to `float32` — convert explicitly when needed.

**Pitfall**
- Aggregation ops expect floats; calling `.mean()` on integer tensors may upcast or error in some versions. Convert with `.float()`.

**Example**
```python
import numpy as np, torch
a = np.arange(3, dtype=np.int64)
t = torch.from_numpy(a)
print(t.dtype)   # torch.int64
print(t.float().mean())  # convert to float before mean
```

---

## 6) Determinism & randomness — beyond `manual_seed`
**What to use**
- `torch.manual_seed(seed)` — CPU RNG.
- `torch.cuda.manual_seed(seed)` — CUDA RNG for current device.
- `torch.cuda.manual_seed_all(seed)` — all CUDA devices.

**For stronger determinism**
```python
import torch
torch.use_deterministic_algorithms(True)    # raise on non-deterministic ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
**Warning:** deterministic algorithms may be slower and some ops may become unavailable.

**Example**
```python
torch.manual_seed(42)
a = torch.rand(3,3)

torch.cuda.manual_seed_all(42)
b = torch.rand(3,3, device='cuda')  # reproducible across runs when seed used
```

---

## 7) GPU Best-Practices & common gotchas
**Moving tensors**
- Use `.to(device)` or `.cuda()` / `.cpu()` to move tensors.
- `pin_memory=True` in DataLoader improves host->device transfer if using non-blocking copies.
- `non_blocking=True` in `.to()` only safe when source is pinned memory.

**Example**
```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = torch.rand(1000)
t_gpu = t.to(device)  # non_blocking=True only when using pinned memory
```

**Device mismatch**
- Operations between CPU and GPU tensors raise runtime errors. Ensure consistent device for all operands.

**NumPy conversion with GPU tensors**
- Move to CPU first: `a.cpu().numpy()`

---

## 8) `argmax` / `argmin` — flattened indices to coordinates
**Default behavior:** returns flattened index (single integer). Convert to 2D coords via `divmod` or `torch.unravel_index` (if available).

**Example**
```python
import torch
res = torch.tensor([[1,9,3],[4,6,2]])
flat_idx = res.argmax()
row, col = divmod(flat_idx.item(), res.shape[1])
print(row, col)

# If torch.unravel_index available:
# coords = torch.unravel_index(flat_idx, res.shape)
```

---

## 9) Minor practical corrections & reminders
- Avoid naming variables `range` because it shadows Python builtin. Use `range_tensor` or `rng`.
- `print(array, array.device)` is invalid for NumPy arrays — `.device` applies to tensors.
- When comparing GPU tensors after `torch.manual_seed`, remember to also use `torch.cuda.manual_seed_all()` if relying on GPU RNG.

---

## 10) Quick function cheatsheet (avoid surprises)
- Copy: `tensor.clone()`
- Force contiguous memory: `tensor.contiguous()`
- Break NumPy/Tensor sharing: `np.copy()` or `tensor.clone()`
- Determinism: `torch.use_deterministic_algorithms(True)` + cuDNN flags
- Convert to float: `tensor.float()` or `tensor.to(torch.float32)`

---

## 11) Small FAQ (common quick answers)
**Q:** Why did my `.view()` fail with "view size is not compatible"?  
**A:** Tensor isn't contiguous; use `.contiguous()` before `.view()` or use `.reshape()`.

**Q:** Why are two identical seeds producing different GPU tensors?  
**A:** You might have set CPU seed but not GPU seed. Use `torch.cuda.manual_seed_all(seed)`.

**Q:** Why does boolean masking return a 1D tensor?  
**A:** Masking selects elements and flattens their output; if you need indices, use `.nonzero()`.

---

### Save & Use
You can save this text as a Markdown file named `advanced_pytorch_topics.md` and open it in VS Code, Obsidian, or Jupyter.

