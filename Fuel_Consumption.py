import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio

# 计算单位离散步长内总等效油耗
def get_fuel(fuel_cost_rate, bat_pwr, SOC_new, delta_SOC, t):
    # 取等效因子
    s = get_EF(delta_SOC)
    # 取SOC惩罚
    K_soc = SOC_punishment(delta_SOC)
    # 计算离散步长内发动机油耗
    fuel_cost = fuel_cost_rate * t # 单位g
    # 计算离散步长内电池能量和等效油耗
    bat_cost = bat_pwr * t # 单位j
    bat2fuel_cost = K_soc * s * (bat_cost / 42600) # 单位g
    # 离散步长内总等效油耗
    eq_fuel_cost = fuel_cost + bat2fuel_cost
    return fuel_cost, eq_fuel_cost

# 建立SOC惩罚函数，使用S型函数
def SOC_punishment(delta_SOC):
    # SOC惩罚函数参数
    a = 1
    b = 0.95
    K_soc = 1 - a * delta_SOC**3 + b * delta_SOC**4 # 曲线参数参考：华南理工大学李晓甫学位论文
    return K_soc

# 建立等效因子
def get_EF(delta_SOC):
    # if delta_SOC > 0 :
    #     s = 2.0
    # else:
    #     s = 0.8
    s = 2.6 # 参考：Self-Adaptive Equivalent Consumption Minimization Strategy for Hybrid Electric Vehicles 常数ECMS方案

    return s