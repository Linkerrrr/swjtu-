# 人工智能 week6 · 矩阵乘法的三种实现

用三种不同层级的引擎实现同一个矩阵乘法 `C = A × B`，并交叉校验结果、对比性能。

- 作者环境：conda 环境 `myenv`，Python 3.11.17
- 显卡：NVIDIA GeForce RTX 5060 Laptop GPU（算力 12.0 / sm_120，Blackwell）
- 依赖版本：NumPy 2.4.6、PyTorch 2.14.1+cu130、matplotlib 3.11.2

---

## 一、原理

$A$ 为 $m \times k$，$B$ 为 $k \times n$，则 $C$ 为 $m \times n$：

$$C_{ij} = \sum_{t=1}^{k} A_{i,t} \cdot B_{t,j}$$

即 **`C` 的第 `i` 行第 `j` 列 = `A` 的第 `i` 行 与 `B` 的第 `j` 列 的点积**。

三种方法算的都是这个公式，区别只在**谁来执行循环**。

| 方法 | 实现 | 循环执行者 |
|---|---|---|
| 一 | 列表推导 + `for` | Python 解释器（逐元素） |
| 二 | `numpy.ndarray @ ndarray` | C / BLAS（CPU） |
| 三 | `torch.Tensor @ Tensor` | CUDA 核心（GPU） |

---

## 二、文件说明

| 文件 | 说明 |
|---|---|
| `矩阵乘法三种实现.ipynb` | **主提交文件**。Jupyter Notebook，已执行，含代码与运行输出 |
| `矩阵乘法三种实现.html` | 同一 notebook 的网页版，双击即可用浏览器查看（无需 Jupyter） |
| `矩阵乘法三种实现.py` | 等价的独立 Python 脚本，命令行直接运行 |
| `运行结果_01_环境与方法一.png` | 运行结果截图：环境信息 + 方法一（含易错点） |
| `运行结果_02_NumPy与PyTorch.png` | 运行结果截图：方法二 NumPy + 方法三 PyTorch(GPU) |
| `运行结果_03_交叉校验与耗时.png` | 运行结果截图：大矩阵三方法交叉校验 + 耗时对比图 |
| `耗时对比图.png` | 单独的耗时对比图（matplotlib 输出） |
| `运行结果_终端输出.txt` | `.py` 脚本的完整控制台输出文本 |

---

## 三、怎么运行

### 方式 1：Jupyter Notebook（推荐）

```bat
conda activate myenv
jupyter notebook "矩阵乘法三种实现.ipynb"
```

然后在网页里 `Kernel → Restart & Run All`。

### 方式 2：命令行脚本

```bat
conda activate myenv
python 矩阵乘法三种实现.py
```

### 方式 3：不想装环境，只看结果

直接双击 `矩阵乘法三种实现.html`，或用图片查看器打开那三张 png。

---

## 四、关键代码

### 方法一：纯 Python

```python
def matmul_listcomp(A, B):
    """外层遍历 A 的行，内层遍历 B 的列（zip(*B) 转置）"""
    return [[sum(a * b for a, b in zip(row, col))
             for col in zip(*B)]
            for row in A]
```

三层结构对应：

1. `sum(a * b for a, b in zip(row, col))` —— 算**一个格子**的点积
2. `for col in zip(*B)` —— 一行配**所有列**
3. `for row in A` —— 所有行都来一遍

### 方法二：NumPy

```python
C = np.array(A) @ np.array(B)
```

### 方法三：PyTorch

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
C = (torch.tensor(A, dtype=torch.float64).to(device)
     @ torch.tensor(B, dtype=torch.float64).to(device))
```

---

## 五、运行结果

小样例：`A` 是 2×3，`B` 是 3×2，结果 2×2。

| 方法 | 结果 |
|---|---|
| 方法一（列表推导 + for） | `[[58, 64], [139, 154]]` |
| 方法二（NumPy） | `[[58, 64], [139, 154]]` |
| 方法三（PyTorch / cuda:0） | `[[58., 64.], [139., 154.]]` |

大矩阵：`(200×300) · (300×400) → 200×400`，结果 80,000 个元素，乘加 24,000,000 次。

```
=== 结果一致性 ===
纯 Python == NumPy   : True
纯 Python == PyTorch : True
NumPy     == PyTorch : True
最大绝对误差         : 0.0
```

性能对比（同一台机器、同一进程内测得）：

| 方法 | 耗时 | 相对纯 Python |
|---|---|---|
| 纯 Python（列表 + for） | 约 3 秒 | 1× |
| NumPy（CPU） | 约 0.028 s | 约 110 倍 |
| PyTorch（GPU / cuda） | 约 0.001 s | 约 2800 倍 |

> 具体数值每次运行会有小幅波动，以 notebook 内的实际输出为准。

---

## 六、结论

1. **数学上完全等价**：三种方法结果逐元素一致，最大绝对误差为 0。
2. **性能差异来自执行层级，不是算法**：
   - 方法一每个乘加都有 Python 解释器开销；
   - 方法二把循环下沉到 C 实现的 BLAS，并利用 CPU 的 SIMD 与缓存；
   - 方法三把循环交给 GPU 上千个 CUDA 核心并行。
3. **形状规则**：`(m×k) · (k×n) = m×n`，中间两个维度必须相等。
4. **易错点**：建结果矩阵必须写 `[[0] * m for _ in range(n)]`；
   写成 `[[0] * m] * n` 会让所有行指向**同一个列表对象**，改一行等于改所有行。
5. 数据规模越小，GPU 优势越不明显（还有显存搬运开销）；规模越大，并行优势越明显。

---

## 七、备注

- 三张 `运行结果_*.png` 的内容全部取自 notebook **真实执行**产生的输出（代码、stdout、matplotlib 图），文字由脚本从 `.ipynb` 中提取后渲染。最权威的记录是 `矩阵乘法三种实现.ipynb` 与 `.html`，里面保留了 Jupyter 原生的执行编号与输出。
- 若课上要求「屏幕截图」，直接双击 `矩阵乘法三种实现.html`，按 `Win + Shift + S` 自行截取即可。
- **关于 `OMP: Error #15`**：conda 的 MKL 版 numpy 与 torch 各自带一份 `libiomp5md.dll`，在同一进程（尤其 Jupyter kernel）里会冲突导致 kernel 直接死掉。本 notebook 的第一个单元格和 `.py` 的开头已经内置了防线：

  ```python
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  ```

  它必须在 `import numpy` / `import torch` **之前**执行。所以直接运行即可，不需要你手动设环境变量；**请不要删除这几行**。
