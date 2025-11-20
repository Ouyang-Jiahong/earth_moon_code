import numpy as np

# IAU2009天文常数表
G = 6.67428e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
mu_earth = 3.986004356e14  # 地球引力常数 (m^3 s^-2)
mu_moon = mu_earth * 0.0123000371  # 月球引力常数 (m^3 s^-2)，根据地球质量比计算

# 来自刘延柱 - 2015 - 关于地月系统的拉格朗日点
d_mean_earth_moon = 3.844e8  # 地月平均距离 (m)

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

# 计算地月系统的总引力参数和角速度
mu_earth_moon = mu_earth + mu_moon  # 总引力参数
omega_c = np.sqrt(mu / d_mean_earth_moon**3)  # 系统的角速度 (rad/s)

# 计算地月质心到地球和月球的距离 a1, a2
# 公式: a1 = m2/(m1+m2) * d, a2 = m1/(m1+m2) * d
# 因为 mu = G*m, 所以质量比 m2/m1 = mu_m/mu_e
mass_ratio = mu_moon / mu_earth
a1 = (mass_ratio / (1 + mass_ratio)) * d_mean_earth_moon  # 质心到地球的距离
a2 = (1 / (1 + mass_ratio)) * d_mean_earth_moon  # 质心到月球的距离
