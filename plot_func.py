import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import mplcursors  # 需要安装: pip install mplcursors
import numpy as np
from scipy.optimize import root_scalar

# ----------------------------
# 1. 全局参数与常量定义
# ----------------------------
from ConstantsAll import *


def compute_lagrange_points(mu_value):
    """Compute approximate positions of the five Lagrange points in the synodic frame.

    Returns an array of shape (5,2) for L1..L5 as (x,y).
    """
    # Primary positions
    x1 = -mu_value
    x2 = 1 - mu_value

    def f(x):
        # derivative of effective potential wrt x (y=0)
        return (
            x
            - (1 - mu_value) * (x + mu_value) / abs(x + mu_value) ** 3
            - mu_value * (x + mu_value - 1) / abs(x + mu_value - 1) ** 3
        )

    def find_root(a, b):
        fa = f(a)
        fb = f(b)
        # Try to expand bracket until sign change (limited attempts)
        attempts = 0
        while fa * fb > 0 and attempts < 60:
            # expand interval outward
            a -= 0.1 * (1 + attempts)
            b += 0.1 * (1 + attempts)
            fa = f(a)
            fb = f(b)
            attempts += 1
        try:
            res = root_scalar(f, bracket=[a, b], method="bisect", xtol=1e-14)
            if res.converged:
                return res.root
        except Exception:
            pass
        # fallback: try secant/newton with a small initial guess
        try:
            res2 = root_scalar(f, x0=(a + b) / 2, method="secant")
            if res2.converged:
                return res2.root
        except Exception:
            return None

    # Collinear points: initial brackets chosen near primaries
    L1_x = find_root(x2 - 0.5, x2 - 1e-9)
    L2_x = find_root(x2 + 1e-9, x2 + 0.5)
    L3_x = find_root(x1 - 0.5, x1 - 1e-9)

    # Triangular points (analytical)
    x_L45 = 0.5 - mu_value
    y_L = np.sqrt(3.0) / 2.0

    L_points = np.zeros((5, 2))
    L_points[0] = [L1_x if L1_x is not None else np.nan, 0.0]
    L_points[1] = [L2_x if L2_x is not None else np.nan, 0.0]
    L_points[2] = [L3_x if L3_x is not None else np.nan, 0.0]
    L_points[3] = [x_L45, y_L]
    L_points[4] = [x_L45, -y_L]

    return L_points


def _plot_trajectories_in_synodic_frame(spacecraft_states, debris_states, t_lead):
    """
    在 CRTBP 同步旋转坐标系中绘制轨迹（两个独立可交互窗口）。

    Parameters:
    - spacecraft_states: (N, 6) 数组，[x, y, z, vx, vy, vz]
    - debris_states: (N, 6) 数组，[x, y, z, vx, vy, vz]
    - t_lead: 机动提前时间（无量纲）
    """
    earth_pos = (-mu, 0)
    moon_pos = (1 - mu, 0)

    # ========================
    # 1. 创建独立的 2D X-Y 平面图
    # ========================
    fig_2d = plt.figure(figsize=(8, 6))
    ax_2d = fig_2d.add_subplot(111)

    line_sc_2d = ax_2d.plot(
        spacecraft_states[:, 0],
        spacecraft_states[:, 1],
        "b-",
        linewidth=2,
        label="Spacecraft",
    )
    line_de_2d = ax_2d.plot(
        debris_states[:, 0], debris_states[:, 1], "r--", linewidth=2, label="Debris"
    )

    ax_2d.plot(earth_pos[0], earth_pos[1], "bo", markersize=12, label="Earth (P1)")
    ax_2d.plot(moon_pos[0], moon_pos[1], "go", markersize=8, label="Moon (P2)")

    # Plot Lagrange points on 2D plot
    colors_collinear = ["m", "c", "y"]
    for i in range(3):
        x_i, y_i = lagrange_points_nomarlized[i][1:3]
        if not np.isnan(x_i):
            ax_2d.plot(
                x_i,
                y_i,
                marker="X",
                color=colors_collinear[i],
                markersize=8,
                label=f"L{i+1}",
            )
            ax_2d.annotate(
                f"L{i+1}", xy=(x_i, y_i), xytext=(5, 5), textcoords="offset points"
            )

    # L4 and L5
    for i in (3, 4):
        x_i, y_i = lagrange_points_nomarlized[i][1:3]
        ax_2d.plot(x_i, y_i, marker="^", color="k", markersize=8, label=f"L{i+1}")
        ax_2d.annotate(
            f"L{i+1}", xy=(x_i, y_i), xytext=(5, 5), textcoords="offset points"
        )

    tca_sc_2d = ax_2d.plot(
        spacecraft_states[-1, 0],
        spacecraft_states[-1, 1],
        "b*",
        markersize=10,
        label="TCA (Spacecraft)",
    )
    tca_de_2d = ax_2d.plot(
        debris_states[-1, 0],
        debris_states[-1, 1],
        "r*",
        markersize=10,
        label="TCA (Debris)",
    )

    ax_2d.set_xlabel("X (Dimensionless)")
    ax_2d.set_ylabel("Y (Dimensionless)")
    ax_2d.set_title(
        f"Trajectories in Synodic Frame (X-Y Plane)\nManeuver Lead Time: {t_lead:.3f} (dimless)"
    )
    ax_2d.legend()
    ax_2d.grid(True, linestyle="--", alpha=0.7)
    ax_2d.axis("equal")

    # 为2D图添加悬停提示
    cursor_2d = mplcursors.cursor(
        [line_sc_2d[0], line_de_2d[0], tca_sc_2d[0], tca_de_2d[0]], hover=True
    )

    @cursor_2d.connect("add")
    def on_add_2d(sel):
        if sel.artist == line_sc_2d[0]:
            idx = int(sel.target.index)
            x, y = spacecraft_states[idx, 0], spacecraft_states[idx, 1]
            z = spacecraft_states[idx, 2]
            sel.annotation.set_text(
                f"SC: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(spacecraft_states) * t_lead:.3f}"
            )
        elif sel.artist == line_de_2d[0]:
            idx = int(sel.target.index)
            x, y = debris_states[idx, 0], debris_states[idx, 1]
            z = debris_states[idx, 2]
            sel.annotation.set_text(
                f"DE: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(debris_states) * t_lead:.3f}"
            )
        elif sel.artist == tca_sc_2d[0]:
            x, y = spacecraft_states[-1, 0], spacecraft_states[-1, 1]
            z = spacecraft_states[-1, 2]
            sel.annotation.set_text(
                f"TCA SC: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {t_lead:.3f}"
            )
        else:
            x, y = debris_states[-1, 0], debris_states[-1, 1]
            z = debris_states[-1, 2]
            sel.annotation.set_text(
                f"TCA DE: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {t_lead:.3f}"
            )

    # ========================
    # 2. 创建独立的 3D 轨迹图
    # ========================
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    line_sc_3d = ax_3d.plot(
        spacecraft_states[:, 0],
        spacecraft_states[:, 1],
        spacecraft_states[:, 2],
        "b-",
        linewidth=2,
        label="Spacecraft",
    )
    line_de_3d = ax_3d.plot(
        debris_states[:, 0],
        debris_states[:, 1],
        debris_states[:, 2],
        "r--",
        linewidth=2,
        label="Debris",
    )

    ax_3d.scatter(
        [earth_pos[0]], [earth_pos[1]], [0], color="blue", s=200, label="Earth (P1)"
    )
    ax_3d.scatter(
        [moon_pos[0]], [moon_pos[1]], [0], color="green", s=100, label="Moon (P2)"
    )

    # Plot Lagrange points on 3D plot (z=0)
    for i in range(5):
        x_i, y_i = lagrange_points_nomarlized[i][1:3]
        if not np.isnan(x_i):
            ax_3d.scatter(
                [x_i],
                [y_i],
                [0],
                marker="X" if i < 3 else "^",
                s=80,
                color=("m" if i == 0 else "c" if i == 1 else "y" if i == 2 else "k"),
                label=f"L{i+1}",
            )

    ax_3d.set_xlabel("X (Dimensionless)")
    ax_3d.set_ylabel("Y (Dimensionless)")
    ax_3d.set_zlabel("Z (Dimensionless)")
    ax_3d.set_title("3D Trajectories in Synodic Frame")
    ax_3d.legend()

    # 为3D图添加悬停提示
    cursor_3d = mplcursors.cursor([line_sc_3d[0], line_de_3d[0]], hover=True)

    @cursor_3d.connect("add")
    def on_add_3d(sel):
        if sel.artist == line_sc_3d[0]:
            idx = int(sel.target.index)
            x, y, z = (
                spacecraft_states[idx, 0],
                spacecraft_states[idx, 1],
                spacecraft_states[idx, 2],
            )
            sel.annotation.set_text(
                f"SC: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(spacecraft_states) * t_lead:.3f}"
            )
        else:
            idx = int(sel.target.index)
            x, y, z = (
                debris_states[idx, 0],
                debris_states[idx, 1],
                debris_states[idx, 2],
            )
            sel.annotation.set_text(
                f"DE: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(debris_states) * t_lead:.3f}"
            )

    # 显示两个独立窗口
    plt.show(block=True)  # 保持窗口打开并响应鼠标事件
    # plt.show()  # 这会同时显示两个 figure
