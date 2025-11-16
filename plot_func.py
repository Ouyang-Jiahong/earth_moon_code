import matplotlib
matplotlib.use('Qt5Agg')  # 必须在 import pyplot 之前设置！
import matplotlib.pyplot as plt
import mplcursors  # 需要安装: pip install mplcursors

# ----------------------------
# 1. 全局参数与常量定义
# ----------------------------
from ConstantsAll import *


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

    line_sc_2d = ax_2d.plot(spacecraft_states[:, 0], spacecraft_states[:, 1], 'b-', linewidth=2, label='Spacecraft')
    line_de_2d = ax_2d.plot(debris_states[:, 0], debris_states[:, 1], 'r--', linewidth=2, label='Debris')

    ax_2d.plot(earth_pos[0], earth_pos[1], 'bo', markersize=12, label='Earth (P1)')
    ax_2d.plot(moon_pos[0], moon_pos[1], 'go', markersize=8, label='Moon (P2)')

    tca_sc_2d = ax_2d.plot(spacecraft_states[-1, 0], spacecraft_states[-1, 1], 'b*', markersize=10,
                           label='TCA (Spacecraft)')
    tca_de_2d = ax_2d.plot(debris_states[-1, 0], debris_states[-1, 1], 'r*', markersize=10, label='TCA (Debris)')

    ax_2d.set_xlabel('X (Dimensionless)')
    ax_2d.set_ylabel('Y (Dimensionless)')
    ax_2d.set_title(f'Trajectories in Synodic Frame (X-Y Plane)\nManeuver Lead Time: {t_lead:.3f} (dimless)')
    ax_2d.legend()
    ax_2d.grid(True, linestyle='--', alpha=0.7)
    ax_2d.axis('equal')

    # 为2D图添加悬停提示
    cursor_2d = mplcursors.cursor([line_sc_2d[0], line_de_2d[0], tca_sc_2d[0], tca_de_2d[0]], hover=True)

    @cursor_2d.connect("add")
    def on_add_2d(sel):
        if sel.artist == line_sc_2d[0]:
            idx = int(sel.target.index)
            x, y = spacecraft_states[idx, 0], spacecraft_states[idx, 1]
            z = spacecraft_states[idx, 2]
            sel.annotation.set_text(
                f"SC: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(spacecraft_states) * t_lead:.3f}")
        elif sel.artist == line_de_2d[0]:
            idx = int(sel.target.index)
            x, y = debris_states[idx, 0], debris_states[idx, 1]
            z = debris_states[idx, 2]
            sel.annotation.set_text(f"DE: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(debris_states) * t_lead:.3f}")
        elif sel.artist == tca_sc_2d[0]:
            x, y = spacecraft_states[-1, 0], spacecraft_states[-1, 1]
            z = spacecraft_states[-1, 2]
            sel.annotation.set_text(f"TCA SC: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {t_lead:.3f}")
        else:
            x, y = debris_states[-1, 0], debris_states[-1, 1]
            z = debris_states[-1, 2]
            sel.annotation.set_text(f"TCA DE: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {t_lead:.3f}")

    # ========================
    # 2. 创建独立的 3D 轨迹图
    # ========================
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    line_sc_3d = ax_3d.plot(spacecraft_states[:, 0], spacecraft_states[:, 1], spacecraft_states[:, 2], 'b-',
                            linewidth=2, label='Spacecraft')
    line_de_3d = ax_3d.plot(debris_states[:, 0], debris_states[:, 1], debris_states[:, 2], 'r--', linewidth=2,
                            label='Debris')

    ax_3d.scatter([earth_pos[0]], [earth_pos[1]], [0], color='blue', s=200, label='Earth (P1)')
    ax_3d.scatter([moon_pos[0]], [moon_pos[1]], [0], color='green', s=100, label='Moon (P2)')

    ax_3d.set_xlabel('X (Dimensionless)')
    ax_3d.set_ylabel('Y (Dimensionless)')
    ax_3d.set_zlabel('Z (Dimensionless)')
    ax_3d.set_title('3D Trajectories in Synodic Frame')
    ax_3d.legend()

    # 为3D图添加悬停提示
    cursor_3d = mplcursors.cursor([line_sc_3d[0], line_de_3d[0]], hover=True)

    @cursor_3d.connect("add")
    def on_add_3d(sel):
        if sel.artist == line_sc_3d[0]:
            idx = int(sel.target.index)
            x, y, z = spacecraft_states[idx, 0], spacecraft_states[idx, 1], spacecraft_states[idx, 2]
            sel.annotation.set_text(
                f"SC: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(spacecraft_states) * t_lead:.3f}")
        else:
            idx = int(sel.target.index)
            x, y, z = debris_states[idx, 0], debris_states[idx, 1], debris_states[idx, 2]
            sel.annotation.set_text(f"DE: ({x:.4f}, {y:.4f}, {z:.4f})\nTime: {idx / len(debris_states) * t_lead:.3f}")

    # 显示两个独立窗口
    plt.show(block=True)  # 保持窗口打开并响应鼠标事件
    # plt.show()  # 这会同时显示两个 figure