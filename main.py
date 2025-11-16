# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from plot_func import _plot_trajectories_in_synodic_frame

# ----------------------------
# 1. 全局参数与常量定义
# ----------------------------
from ConstantsAll import *

# ----------------------------
# 2. CRTBP 动力学函数
# ----------------------------
def crtbp_dynamics(t, state):
    """计算 CRTBP 动力学导数 (无控)"""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    # 避免除零错误
    if r1 < 1e-12:
        r1 = 1e-12
    if r2 < 1e-12:
        r2 = 1e-12

    gx = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x + mu - 1) / r2 ** 3
    gy = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
    gz = - (1 - mu) * z / r1 ** 3 - mu * z / r2 ** 3

    hx = 2 * vy
    hy = -2 * vx
    hz = 0.0

    ax = gx + hx
    ay = gy + hy
    az = gz + hz
    return np.array([vx, vy, vz, ax, ay, az])


def crtbp_with_thrust(t, state, u, alpha):
    """计算带推力的 CRTBP 动力学导数 (Eq. 2)"""
    x, y, z, vx, vy, vz, m = state
    # 仅提取前6个状态（位置和速度）用于无控动力学
    state_6d = np.array([x, y, z, vx, vy, vz])
    _, _, _, ax, ay, az = crtbp_dynamics(t, state_6d)

    # 添加推力项
    ax += u * T_max * alpha[0] / m
    ay += u * T_max * alpha[1] / m
    az += u * T_max * alpha[2] / m

    # 质量导数 (标量)
    dm_dt = - u * T_max / c

    return np.array([vx, vy, vz, ax, ay, az, dm_dt])

# ----------------------------
# 3. B平面与碰撞概率相关函数
# ----------------------------
def compute_B_plane_rotation(rs_tca, vs_tca, rp_tca, vp_tca):
    """根据 Eq. 6 计算从 ECI 到 B 平面的旋转矩阵 R_b"""
    # ... (计算 ûξ, ûη, ûζ 并组成矩阵)
    return R_b

def collision_probability(rf, rs_tca, R_b, S_A, sigma):
    """根据 Eq. 15 计算碰撞概率 P_C"""
    diff = rf - rs_tca
    proj = R_b @ diff
    xi, _, zeta = proj  # η 方向被 S 矩阵丢弃
    d_squared = xi**2 + zeta**2
    exponent = np.log(1 - np.exp(-S_A**2 / (2 * sigma**2))) - d_squared / (2 * sigma**2)
    return np.exp(exponent)

# ----------------------------
# 4. 最优控制核心：同伦打靶法求解器
# ----------------------------
class CollisionAvoidanceOptimizer:
    def __init__(self, r0, v0, m0, rs_t0, t0, tf, W):
        self.r0, self.v0, self.m0 = r0, v0, m0
        self.rs_t0 = rs_t0  # 碎片在 t0 的状态
        self.t0, self.tf = t0, tf
        self.W = W
        self.R_b = None  # 需在初始化时计算

    def _dynamics_and_costate(self, t, y, eps, lambda0):
        """同时积分状态和协态方程 (Eq. 2 + Eq. 18)"""
        # y = [r, v, m, lambda_r, lambda_v, lambda_m]
        # 1. 从 y 中提取状态和协态
        # 2. 计算推力方向 alpha = -lambda_v / ||lambda_v||
        # 3. 计算切换函数 S_star (Eq. 21)
        # 4. 根据 eps 和 S_star 计算 u (Eq. 22)
        # 5. 计算状态导数 (Eq. 2)
        # 6. 计算协态导数 (Eq. 18)
        # 7. 返回 [state_dot, costate_dot]
        pass

    def _boundary_residual(self, lambda_guess, eps):
        """计算边界条件残差 (Eq. 25)，用于打靶法"""
        lambda_r0, lambda_v0, lambda_m0, lambda0 = lambda_guess
        y0 = np.hstack([self.r0, self.v0, self.m0, lambda_r0, lambda_v0, lambda_m0])

        # 积分整个状态-协态系统
        sol = solve_ivp(
            fun=lambda t, y: self._dynamics_and_costate(t, y, eps, lambda0),
            t_span=(self.t0, self.tf),
            y0=y0,
            method='LSODA'
        )

        # 提取终端状态
        yf = sol.y[:, -1]
        rf = yf[0:3]
        lambda_rf = yf[6:9]
        lambda_vf = yf[9:12]
        lambda_mf = yf[12]

        # 计算期望的终端协态 (Eq. 24)
        expected_lambda_rf = ... # 基于 collision_probability 的梯度
        expected_lambda_vf = np.zeros(3)
        expected_lambda_mf = 0.0

        # 返回残差
        return np.hstack([
            lambda_rf - expected_lambda_rf,
            lambda_vf - expected_lambda_vf,
            lambda_mf - expected_lambda_mf
        ])

    def solve_homotopy(self, initial_lambda_guess):
        """执行同伦法，从 eps=1 到 eps=0"""
        current_lambda_guess = initial_lambda_guess
        eps = 1.0
        while eps >= 0:
            # 使用打靶法求解当前 eps 下的 TPBVP
            result = root(
                fun=self._boundary_residual,
                x0=current_lambda_guess,
                args=(eps,)
            )
            if not result.success:
                raise RuntimeError(f"Shooting method failed at eps={eps}")

            current_lambda_guess = result.x
            eps = max(0, eps - 0.1) # 逐步减小 eps

        # 最终解在 result.x 中
        return result

# ----------------------------
# 5. 主程序流程 (对应 Table 1)
# ----------------------------
def main():
    # Step 1: 输入 TCA 时刻状态 (来自 Table 2)
    # 注意：论文中状态是无量纲的
    # （已在文件开头引入）
    # r_pc_tca = np.array([0.9708918431, 0.08693169497, -0.0141139994])
    # v_pc_tca = np.array([0.1486521434, -0.2247801445, 0.3899588750])
    # r_sc_tca = np.array([0.9708920000, 0.08693169497, -0.0141139994])
    # v_sc_tca = np.array([0.148652143405387, 0.1500000000, -0.3100000000])

    # Step 2: 从 TCA 时刻反向积分，得到 t0 时刻状态
    # 设定机动提前时间为 3 小时
    maneuver_lead_time = 3 * 3600  # 秒，但我们需要无量纲时间

    # CRTBP 无量纲时间转换
    # T_dimless = T_dim / T_char, 其中 T_char = 1 / n, n = sqrt(G(M1+M2)/R^3)
    # 对于地月系统，1 无量纲时间单位 ≈ 1 / (2*np.pi) 个会合周期 ≈ 3.73 天
    # 因此，3小时 ≈ 3 / (24 * 3.73) ≈ 0.0335 无量纲单位
    # 为了简化，我们直接使用论文中的值，通常仿真中直接设为 0.01 ~ 0.1
    # t_lead_dimless = 0.04  # 这是一个近似值，需根据实际情况校准
    t_lead_dimless = 5

    # 航天器初始状态（无控反向积分）
    initial_state_pc = np.hstack([r_pc_tca, v_pc_tca])
    sol_pc = solve_ivp(
        crtbp_dynamics,
        t_span=(0, -t_lead_dimless),  # 反向积分
        y0=initial_state_pc,
        method='LSODA',
        rtol=1e-12,
        atol=1e-12
    )
    r0, v0 = sol_pc.y[:3, -1], sol_pc.y[3:, -1]

    # 碎片初始状态（无控反向积分）
    initial_state_sc = np.hstack([r_sc_tca, v_sc_tca])
    sol_sc = solve_ivp(
        crtbp_dynamics,
        t_span=(0, -t_lead_dimless),
        y0=initial_state_sc,
        method='LSODA',
        rtol=1e-12,
        atol=1e-12
    )
    r_sc_t0, v_sc_t0 = sol_sc.y[:3, -1], sol_sc.y[3:, -1]

    print(f"t0 时刻航天器状态: r={r0}, v={v0}")
    print(f"t0 时刻碎片状态: r={r_sc_t0}, v={v_sc_t0}")

    # ----------------------------
    # Step 6: 正向积分，获取完整轨迹（用于绘图）
    # ----------------------------
    t_eval = np.linspace(0, t_lead_dimless, 300)  # 生成时间点用于绘图

    # 航天器轨迹（无控）
    sol_pc_forward = solve_ivp(
        crtbp_dynamics,
        t_span=(0, t_lead_dimless),
        y0=initial_state_pc,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-12,
        atol=1e-12
    )
    spacecraft_states = sol_pc_forward.y.T  # Shape: (N, 6)

    # 碎片轨迹（无控）
    sol_sc_forward = solve_ivp(
        crtbp_dynamics,
        t_span=(0, t_lead_dimless),
        y0=initial_state_sc,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-12,
        atol=1e-12
    )
    debris_states = sol_sc_forward.y.T  # Shape: (N, 6)

    # ----------------------------
    # Step 7: 绘制轨迹
    # ----------------------------
    _plot_trajectories_in_synodic_frame(spacecraft_states, debris_states, t_lead_dimless)

    # Step 3: 初始化参数
    W = 10.0
    t0 = tf_tca - 3 * 3600  # 例如，提前3小时
    tf = tf_tca

    # Step 4 & 5: 同伦打靶法求解
    optimizer = CollisionAvoidanceOptimizer(r0, v0, 1.0, rs_t0, t0, tf, W)
    initial_guess = np.random.rand(10) * 0.1  # 随机初始猜测
    solution = optimizer.solve_homotopy(initial_guess)

    # Step 6: 检查碰撞概率
    P_c = collision_probability(rf_final, r_sc_tca, R_b, S_A, sigma)
    if P_c < 1e-4:
        print("CAM 成功！")
    else:
        # Step 7: 调整 W 并重试
        W *= 2
        # ... 递归或循环调用 optimizer

    # Step 8: 可视化结果
    # ... 绘制轨迹、推力曲线、质量变化

if __name__ == "__main__":
    main()