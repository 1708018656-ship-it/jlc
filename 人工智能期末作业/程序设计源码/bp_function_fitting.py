# -*- coding: utf-8 -*-
"""
BP 神经网络二维函数拟合与可视化
================================

目标函数:  f(x, y) = sin(x) * cos(y)
变量范围:  x in [-3, 3], y in [-3, 3]

本程序使用 PyTorch 搭建一个多层感知机(BP 神经网络),
对上述二维函数进行回归拟合, 并完成以下工作:
    1. 随机生成样本数据, 并按 70%/30% 划分训练集与测试集;
    2. 构建 "输入层-隐藏层-隐藏层-输出层" 的 BP 神经网络;
    3. 使用均方误差(MSE)损失与 Adam 优化器进行训练, 记录每轮 loss;
    4. 绘制训练过程的 loss 变化曲线;
    5. 在网格点上分别计算真实函数值与网络预测值, 绘制三维曲面对比图;
    6. 输出测试集上的误差指标(MSE / RMSE / MAE / R^2), 分析拟合效果。

运行环境: Python 3.10+ , 依赖 numpy / matplotlib / torch
作者: (请填写 学号-姓名)
"""

import os
import numpy as np
import matplotlib

# 使用非交互式后端, 便于在无显示环境中直接保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 3d 投影)

import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# 0. 全局设置: 随机种子 / 设备 / 输出目录
# ----------------------------------------------------------------------
SEED = 42                       # 固定随机种子, 保证结果可复现
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # 图片保存目录(脚本所在目录)

# 变量取值范围(与目标函数定义一致)
X_MIN, X_MAX = -3.0, 3.0
Y_MIN, Y_MAX = -3.0, 3.0


def target_function(x, y):
    """目标函数 f(x, y) = sin(x) * cos(y)。支持 numpy 数组运算。"""
    return np.sin(x) * np.cos(y)


# ----------------------------------------------------------------------
# 1. 数据生成与划分
# ----------------------------------------------------------------------
def generate_dataset(n_samples=1000, test_ratio=0.3):
    """在给定区间内随机生成 n_samples 个样本点, 并划分训练/测试集。

    返回: (x_train, y_train, x_test, y_test), 均为 numpy 数组。
        输入特征维度为 2 (x, y), 输出维度为 1 (函数值)。
    """
    # 在 [X_MIN, X_MAX] x [Y_MIN, Y_MAX] 区间内均匀随机采样
    x = np.random.uniform(X_MIN, X_MAX, size=(n_samples, 1))
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n_samples, 1))
    inputs = np.hstack([x, y]).astype(np.float32)          # 形状 (N, 2)
    outputs = target_function(x, y).astype(np.float32)     # 形状 (N, 1)

    # 随机打乱后按比例划分训练集与测试集
    idx = np.random.permutation(n_samples)
    n_test = int(n_samples * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    return (inputs[train_idx], outputs[train_idx],
            inputs[test_idx], outputs[test_idx])


# ----------------------------------------------------------------------
# 2. BP 神经网络结构定义
# ----------------------------------------------------------------------
class BPNet(nn.Module):
    """简单的 BP 神经网络(多层感知机)。

    结构: 输入层(2) - 隐藏层1(64) - 隐藏层2(64) - 输出层(1)
    隐藏层使用 Tanh 激活函数, 因为目标函数光滑且取值有正有负,
    Tanh 能较好地逼近这类平滑非线性函数。
    """

    def __init__(self, in_dim=2, hidden=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 3. 模型训练
# ----------------------------------------------------------------------
def train_model(x_train, y_train, epochs=400, lr=1e-3):
    """训练 BP 网络, 返回训练好的模型与每轮的损失列表。"""
    model = BPNet().to(DEVICE)
    criterion = nn.MSELoss()                               # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 转换为张量并放到目标设备
    xt = torch.from_numpy(x_train).to(DEVICE)
    yt = torch.from_numpy(y_train).to(DEVICE)

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(xt)                                   # 前向传播
        loss = criterion(pred, yt)
        loss.backward()                                    # 反向传播(BP 核心)
        optimizer.step()                                   # 更新权重
        loss_history.append(loss.item())

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} - train MSE: {loss.item():.6f}")

    return model, loss_history


# ----------------------------------------------------------------------
# 4. 模型评估
# ----------------------------------------------------------------------
def evaluate(model, x_test, y_test):
    """在测试集上计算 MSE / RMSE / MAE / R^2 指标。"""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test).to(DEVICE)).cpu().numpy()

    err = pred - y_test
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ----------------------------------------------------------------------
# 5. 可视化: 训练 loss 曲线
# ----------------------------------------------------------------------
def plot_loss(loss_history, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history,
             color="#1f77b4", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss Curve  f(x,y)=sin(x)cos(y)")
    plt.yscale("log")                 # 对数坐标更清楚地展示收敛过程
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] {save_path}")


# ----------------------------------------------------------------------
# 6. 可视化: 真实曲面 vs 网络拟合曲面
# ----------------------------------------------------------------------
def plot_surface_compare(model, save_path, grid_n=80):
    # 在测试范围内生成规则网格
    gx = np.linspace(X_MIN, X_MAX, grid_n)
    gy = np.linspace(Y_MIN, Y_MAX, grid_n)
    GX, GY = np.meshgrid(gx, gy)

    Z_true = target_function(GX, GY)                       # 真实函数值

    grid_in = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        Z_pred = model(torch.from_numpy(grid_in).to(DEVICE)).cpu().numpy()
    Z_pred = Z_pred.reshape(GX.shape)                      # 网络预测值

    fig = plt.figure(figsize=(15, 5))

    # (1) 原函数三维曲面
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(GX, GY, Z_true, cmap="viridis")
    ax1.set_title("True Function")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f")

    # (2) 神经网络拟合曲面
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(GX, GY, Z_pred, cmap="plasma")
    ax2.set_title("BP Network Prediction")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f")

    # (3) 绝对误差曲面
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(GX, GY, np.abs(Z_pred - Z_true), cmap="inferno")
    ax3.set_title("Absolute Error |pred - true|")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("error")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[saved] {save_path}")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("Target function: f(x, y) = sin(x) * cos(y),  x,y in [-3, 3]")

    # 1) 数据生成与划分
    x_train, y_train, x_test, y_test = generate_dataset(n_samples=1000,
                                                        test_ratio=0.3)
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    # 2) 训练模型
    model, loss_history = train_model(x_train, y_train, epochs=400, lr=1e-3)

    # 3) 评估
    metrics = evaluate(model, x_test, y_test)
    print("\n==== Test set metrics ====")
    for k, v in metrics.items():
        print(f"{k:>5s}: {v:.6f}")

    # 4) 可视化
    plot_loss(loss_history, os.path.join(OUT_DIR, "loss_curve.png"))
    plot_surface_compare(model, os.path.join(OUT_DIR, "fit_3d_compare.png"))

    print("\nDone. Figures saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
