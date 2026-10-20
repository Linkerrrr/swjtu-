# -*- coding: utf-8 -*-
"""
人工智能 week6 —— 矩阵乘法的三种实现

    方法一：纯 Python（列表推导 + for 循环）
    方法二：NumPy 向量化（CPU）
    方法三：PyTorch 张量（GPU / CUDA）

运行：
    conda activate myenv
    python matrix_multiplication.py

三种方法算的是同一个公式
    C[i][j] = sum over t of  A[i][t] * B[t][j]
因此结果必须逐元素完全一致。
"""
import os
import sys
import time

# conda 的 MKL 版 numpy 与 torch 各自带一份 libiomp5md.dll，
# 在同一个进程里会冲突并报 OMP: Error #15。
# 这一行必须放在 import numpy / torch 之前。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")          # 无窗口后端，只存图不弹窗
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# 方法一：纯 Python
# ======================================================================
def matmul_listcomp(A, B):
    """列表推导 + for：外层遍历 A 的行，内层遍历 B 的列（zip(*B) 转置）"""
    return [[sum(a * b for a, b in zip(row, col))
             for col in zip(*B)]
            for row in A]


def matmul_loops(A, B):
    """等价写法：三重 for 循环"""
    n, m, k = len(A), len(B[0]), len(B)
    C = [[0] * m for _ in range(n)]        # 注意：不能写 [[0] * m] * n
    for i in range(n):
        for j in range(m):
            s = 0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


# ======================================================================
# 主程序
# ======================================================================
def main():
    print("=" * 62)
    print("人工智能 week6 · 矩阵乘法的三种实现")
    print("=" * 62)
    print("Python   :", sys.version.split()[0])
    print("解释器   :", sys.executable)
    print("NumPy    :", np.__version__)
    print("PyTorch  :", torch.__version__)
    print("CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("显卡     :", torch.cuda.get_device_name(0))
        print("算力     :", torch.cuda.get_device_capability(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备 :", device)

    # ---------------------------------------------------------------- 小例子
    A = [[1, 2, 3],
         [4, 5, 6]]            # 2 行 3 列
    B = [[7, 8],
         [9, 10],
         [11, 12]]             # 3 行 2 列

    print("\n" + "-" * 62)
    print("【小样例】A 是 2x3，B 是 3x2")
    print("-" * 62)
    print("A =", A)
    print("B =", B)
    print("zip(*B) 取 B 的列 :", list(zip(*B)))
    print()
    print("方法一 列表推导 + for :", matmul_listcomp(A, B))
    print("方法一 三重 for 循环  :", matmul_loops(A, B))
    print("方法二 NumPy (@)      :", (np.array(A) @ np.array(B)).tolist())
    print("方法三 PyTorch (@)    :",
          (torch.tensor(A, dtype=torch.float64).to(device)
           @ torch.tensor(B, dtype=torch.float64).to(device)).cpu().numpy().tolist())

    # ---------------------------------------------------------------- 易错点
    print("\n" + "-" * 62)
    print("【易错点】建结果矩阵")
    print("-" * 62)
    wrong = [[0] * 2] * 2
    wrong[0][0] = 99
    print("[[0]*2]*2                 →", wrong,
          "| 两行是同一对象:", wrong[0] is wrong[1])
    right = [[0] * 2 for _ in range(2)]
    right[0][0] = 99
    print("[[0]*2 for _ in range(2)] →", right)

    # ---------------------------------------------------------------- 大矩阵
    M, K, N = 200, 300, 400
    rng = np.random.default_rng(2026)
    Al = rng.integers(-50, 50, size=(M, K)).tolist()
    Bl = rng.integers(-50, 50, size=(K, N)).tolist()

    print("\n" + "-" * 62)
    print(f"【大矩阵】({M}x{K}) · ({K}x{N}) → {M}x{N}")
    print(f"结果 {M*N:,} 个元素，乘加 {M*K*N:,} 次")
    print("-" * 62)

    t0 = time.perf_counter()
    C1 = matmul_listcomp(Al, Bl)
    t1 = time.perf_counter()

    An, Bn = np.array(Al), np.array(Bl)
    t2 = time.perf_counter()
    C2 = An @ Bn
    t3 = time.perf_counter()

    Ag = torch.tensor(Al, dtype=torch.float64).to(device)
    Bg = torch.tensor(Bl, dtype=torch.float64).to(device)
    _ = Ag @ Bg                                  # 预热
    if device == "cuda":
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    C3 = Ag @ Bg
    if device == "cuda":
        torch.cuda.synchronize()
    t5 = time.perf_counter()

    C1n = np.array(C1, dtype=np.float64)
    C3n = C3.cpu().numpy()

    print("纯 Python == NumPy   :", np.array_equal(C1n, C2))
    print("纯 Python == PyTorch :", np.array_equal(C1n, C3n))
    print("NumPy     == PyTorch :", np.array_equal(C2, C3n))
    print("最大绝对误差         :", np.abs(C1n - C3n).max())
    print()
    print("结果左上角 2x3 :")
    print(C1n[:2, :3])
    print("结果右下角 2x3 :")
    print(C1n[-2:, -3:])

    t_list, t_np, t_torch = t1 - t0, t3 - t2, t5 - t4
    print("\n=== 耗时对比 ===")
    print(f"方法一 纯 Python      : {t_list:10.4f} s")
    print(f"方法二 NumPy (CPU)    : {t_np:10.6f} s    加速 {t_list/t_np:,.0f} 倍")
    print(f"方法三 PyTorch ({device:4s}) : {t_torch:10.6f} s    加速 {t_list/t_torch:,.0f} 倍")

    # ---------------------------------------------------------------- 画图
    labels = [f"纯 Python\n(列表+for)", "NumPy\n(CPU)", f"PyTorch\n({device.upper()})"]
    times = [t_list, t_np, t_torch]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=120)
    bars = ax.barh(labels, times, color=["#c41e3a", "#2e7d32", "#1565c0"])
    ax.set_xscale("log")
    ax.set_xlabel("耗时（秒，对数坐标）")
    ax.set_title(f"矩阵乘法耗时对比   ({M}x{K}) · ({K}x{N})")
    for bar, t in zip(bars, times):
        ax.text(t * 1.2, bar.get_y() + bar.get_height() / 2,
                f"{t:.5f} s", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    out_png = os.path.join(HERE, "耗时对比图.png")
    plt.savefig(out_png)
    print(f"\n耗时对比图已保存: {out_png}")
    print("=" * 62)


if __name__ == "__main__":
    main()
