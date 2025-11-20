import numpy as np
from scipy.optimize import fsolve

# IAU2009天文常数表
G = 6.67428e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
mu_earth = 3.986004356e14  # 地球引力常数 (m^3 s^-2)
mu_moon = mu_earth * 0.0123000371  # 月球引力常数 (m^3 s^-2)，根据地球质量比计算

# 来自刘延柱 - 2015 - 关于地月系统的拉格朗日点
d_mean_earth_moon = 3.844e8  # 地月平均距离 (m)

# 计算地月系统的总引力参数和角速度
mu_earth_moon = mu_earth + mu_moon  # 总引力参数
omega_c = np.sqrt(mu_earth_moon / d_mean_earth_moon**3)  # 系统的角速度 (rad/s)

# 计算地月质心到地球和月球的距离 a1, a2
# 公式: a1 = m2/(m1+m2) * d, a2 = m1/(m1+m2) * d
# 因为 mu = G*m, 所以质量比 m2/m1 = mu_m/mu_e
mass_ratio = mu_moon / mu_earth
a1 = (mass_ratio / (1 + mass_ratio)) * d_mean_earth_moon  # 质心到地球的距离
a2 = (1 / (1 + mass_ratio)) * d_mean_earth_moon  # 质心到月球的距离


def lagrange_equation(x, y=0):
    """
    拉格朗日点的平衡方程 F1 + F2 + Fc = 0 在 (O-xy) 坐标系下的分量形式。
    此函数返回方程组的残差值。
    公式来自论文：刘延柱 - 2015 - 关于地月系统的拉格朗日点
    """
    # 计算飞船P相对于地球O1和月球O2的矢径
    rho1 = np.sqrt((x + a1) ** 2 + y**2)
    rho2 = np.sqrt((x - a2) ** 2 + y**2)

    # 方程(8a) 和 (8b) 的残差
    eq_x = mu_earth * (x + a1) / rho1**3 + mu_moon * (x - a2) / rho2**3 - omega_c**2 * x
    eq_y = mu_earth * y / rho1**3 + mu_moon * y / rho2**3 - omega_c**2 * y

    return [eq_x, eq_y]


# --- 求解 L1, L2, L3 (y=0, 在x轴上) ---
# 定义一个只求解x轴上平衡点的函数
def lagrange_equation_x(x):
    """当 y=0 时，仅返回方程(8a)的残差"""
    # 由于y=0, rho1 = |x+a1|, rho2 = |x-a2|
    # 需要特别注意符号，因为x可能在不同的区间
    rho1 = abs(x + a1)
    rho2 = abs(x - a2)

    # 为了避免除零错误，对非常小的rho进行处理
    if rho1 < 1e-10:
        rho1 = 1e-10
    if rho2 < 1e-10:
        rho2 = 1e-10

    eq_x = mu_earth * (x + a1) / rho1**3 + mu_moon * (x - a2) / rho2**3 - omega_c**2 * x
    return eq_x


# 为每个点提供初始猜测值和搜索区间
# L1: 位于地球和月球之间 (-a1, a2)
x_L1_initial_guess = 0.5 * (a2 - a1)  # 中间附近
L1_solution = fsolve(lagrange_equation_x, x_L1_initial_guess)
L1_x = L1_solution[0]
L1_y = 0.0

# L2: 位于月球外侧 (a2, +∞)
x_L2_initial_guess = a2 + 0.1 * d_mean_earth_moon  # 月球外侧一点
L2_solution = fsolve(lagrange_equation_x, x_L2_initial_guess)
L2_x = L2_solution[0]
L2_y = 0.0

# L3: 位于地球外侧 (-∞, -a1)
x_L3_initial_guess = -a1 - 0.1 * d_mean_earth_moon  # 地球外侧一点
L3_solution = fsolve(lagrange_equation_x, x_L3_initial_guess)
L3_x = L3_solution[0]
L3_y = 0.0

# --- 求解 L4, L5 (等边三角形) ---
# 根据论文（刘延柱 - 2015 - 关于地月系统的拉格朗日点）推导，L4和L5与地球O1和月球O2构成边长为d的等边三角形。
# 在(O-xy)坐标系中，O1位于(-a1, 0), O2位于(a2, 0)
# 等边三角形的第三个顶点坐标可以通过几何关系计算。

# L4: 位于x轴上方
L4_x = (a2 - a1) / 2  # x坐标是O1和O2的中点
L4_y = np.sqrt(3) / 2 * d_mean_earth_moon  # y坐标是等边三角形的高

# L5: 位于x轴下方
L5_x = (a2 - a1) / 2  # x坐标与L4相同
L5_y = -np.sqrt(3) / 2 * d_mean_earth_moon  # y坐标为负

# 将结果整理成列表以便打印
# 通过与论文中的数值对比，确认计算结果的正确性
lagrange_points = [
    ("L1", L1_x, L1_y),
    ("L2", L2_x, L2_y),
    ("L3", L3_x, L3_y),
    ("L4", L4_x, L4_y),
    ("L5", L5_x, L5_y),
]

lagrange_points_nomarlized = [
    (name, x / d_mean_earth_moon, y / d_mean_earth_moon)
    for name, x, y in lagrange_points
]


# 来自Tu 等 - 2025 - Optimal control for low-thrust collision avoidance in CRTBP
mu = 0.01215  # 地月系统无量纲质量参数
T_max = 50e-3  # 最大推力 (N)
I_sp = 3800  # 比冲 (s)
g0 = 9.81  # 重力加速度 (m/s^2)
c = I_sp * g0  # 排气速度 (m/s)

# TCA时刻状态 (来自论文 Table 2)
r_pc_tca = np.array([0.9708918431, 0.08693169497, -0.0141139994])
v_pc_tca = np.array([0.1486521434, -0.2247801445, 0.3899588750])
r_sc_tca = np.array([0.9708920000, 0.08693169497, -0.0141139994])
v_sc_tca = np.array([0.148652143405387, 0.1500000000, -0.3100000000])

S_A = 25 + 4.7  # 等效半径之和 (km)
C_b = np.array([[2.686552e-2, 0], [0, 7.205562e-2]])  # B平面协方差矩阵
sigma = np.sqrt(C_b[0, 0] * C_b[1, 1])
