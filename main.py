# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

# ----------------------------
# 1. 全局参数与常量定义
# ----------------------------
mu = 0.01215  # 地月系统无量纲质量参数
T_max = 50e-3  # 最大推力 (N)
I_sp = 3800    # 比冲 (s)
g0 = 9.81      # 重力加速度 (m/s^2)
c = I_sp * g0  # 排气速度 (m/s)

# TCA时刻状态 (来自论文 Table 2)
r_pc_tca = np.array([0.9708918431, 0.08693169497, -0.0141139994])
v_pc_tca = np.array([0.1486521434, -0.2247801445, 0.3899588750])
r_sc_tca = np.array([0.9708920000, 0.08693169497, -0.0141139994])
v_sc_tca = np.array([0.148652143405387, 0.1500000000, -0.3100000000])

S_A = 25 + 4.7  # 等效半径之和 (km)
C_b = np.array([[2.686552e-2, 0], [0, 7.205562e-2]])  # B平面协方差矩阵
sigma = np.sqrt(C_b[0, 0] * C_b[1, 1])

# ----------------------------
# 2. CRTBP 动力学函数
# ----------------------------
def crtbp_dynamics(t, state):
    """计算 CRTBP 动力学导数 (无控)"""
    x, y, z, vx, vy, vz = state
    # ... (根据 Eq. 3 & 4 实现)
    return np.array([vx, vy, vz, ax, ay, az])

def crtbp_with_thrust(t, state, u, alpha):
    """计算带推力的 CRTBP 动力学导数 (Eq. 2)"""
    x, y, z, vx, vy, vz, m = state
    # ... (调用 crtbp_dynamics 并加上推力项)
    return derivatives

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
    # Step 1: 输入 TCA 时刻状态
    # (已定义在全局变量中)

    # Step 2: 从 TCA 时刻反向积分，得到 t0 时刻状态
    # ... 使用 crtbp_dynamics 反向积分航天器和碎片

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