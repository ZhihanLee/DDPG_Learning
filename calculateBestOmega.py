# -*- coding: utf-8 -*-
"""
本程序用于计算指定初始车速下，Prius的发动机、MG1、MG2转速
用于给仿真创造初始条件
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio

def calculate_speed(car_spd):
    # 基本参数
    Wheel_R = 0.287
    mass = 1449
    C_roll  = 0.013
    density_air = 1.2
    area_frontal = 2.23
    G = 9.81
    C_d = 0.26
    # the factor of F_roll
    T_factor = 0.015
    
    # paramsmeters of transmission system
    # number of teeth
    Pgs_R = 2.6                        # 齿圈齿数
    Pgs_S = 1                        # 太阳轮齿数
    # speed ratio from ring gear to wheel  从齿圈到车轮的传动比(主减速器传动比)
    Pgs_K = 3.93  

    # # The optimal curve of engine   
    # Eng_pwr_opt_list = np.arange(0, 57, 1) * 1000                                                                                             # 发动机功率范围
    # W_list = [91.106186954104, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 89.0117918517108, 94.24777960769379, 117.28612573401894, 121.47491593880532, 132.9940890019679, 142.4188669627373, 145.56045961632708, 168.59880574265222, 170.69320084504542, 181.1651763570114, 183.25957145940458, 183.25957145940458, 192.684349420174, 204.20352248333657, 210.48670779051614, 215.72269554649912, 220.95868330248211, 248.18581963359367, 237.71384412162766, 255.51620249196986, 260.7521902479528, 264.9409804527392, 269.1297706575256, 274.3657584135086, 281.69614127188476, 289.026524130261, 294.262511886244, 298.4513020910303, 301.59289474462014, 305.78168494940655, 309.9704751541929, 314.1592653589793, 320.4424506661589, 328.8200310757317, 336.15041393410786, 344.52799434368063, 352.90557475325346, 361.2831551628262, 369.660735572399, 378.0383159819718, 385.368698840348, 394.7934768011173, 402.12385965949346, 410.5014400690663, 418.87902047863906, 427.2566008882119, 436.6813788489813, 442.96456415616086, 451.3421445657336, 459.7197249753064]
    # Eng_pwr_func = interp1d(Eng_pwr_opt_list, W_list)                                                                               # 一维插值  变成函数，得到功率和W曲线 W是角速度
    # Eng_spd_list = np.arange(0, 4501, 125) * (2 * math.pi) / 60
    # Eng_spd_list = Eng_spd_list[np.newaxis, :]
    # Eng_trq_list = np.arange(0, 111, 5) * (121 / 110)
    # Eng_trq_list = Eng_trq_list[np.newaxis, :]
    # data_path = 'Eng_bsfc_map.mat'
    # data = scio.loadmat(data_path)
    # Eng_bsfc_map = data['Eng_bsfc_map']    
    # Eng_trq_maxP = [-4.1757e-009, 6.2173e-006, -3.4870e-003, 9.1743e-001, 2.0158e+001]                                                                   #多项式系数数组
    # Eng_fuel_map = Eng_bsfc_map * (Eng_spd_list.T * Eng_trq_list) / 3600 / 1000
    
    # # fuel consumption (g)
    # Eng_fuel_func = interp2d(Eng_trq_list, Eng_spd_list, Eng_fuel_map)                                                                                   #变成一个函数 可以用转矩和转速作为输入参数输出油耗

    # # Motor
    # # motor speed list (rad/s)
    # Mot_spd_list = np.arange(-6000, 6001, 200) * (2 * math.pi) / 60        
    # # motor torque list (Nm)
    # Mot_trq_list = np.arange(-400, 401, 10)
    
    # # motor efficiency map
    # data_path1 = 'Mot_eta_quarter.mat'
    # data1 = scio.loadmat(data_path1)
    # Mot_eta_quarter = data1['Mot_eta_quarter']
    # Mot_eta_alltrqs = np.concatenate(([np.fliplr(Mot_eta_quarter[:, 1:]), Mot_eta_quarter]), axis = 1)
    # Mot_eta_map = np.concatenate(([np.flipud(Mot_eta_alltrqs[1:, :]), Mot_eta_alltrqs]))
    # # motor efficiency
    # Mot_eta_map_func = interp2d(Mot_trq_list, Mot_spd_list, Mot_eta_map)                                                                                 #变成一个函数 可以用转矩和转速作为输入参数输出油耗             
    
    # #  motor maximum torque
    # Mot_trq_max_quarter = np.array([400,400,400,400,400,400,400,347.200000000000,297.800000000000,269.400000000000,241,221.800000000000,202.600000000000,186.400000000000,173.200000000000,160,148,136,126.200000000000,118.600000000000,111,105.800000000000,100.600000000000,96.2000000000000,92.6000000000000,89,87.4000000000000,85.8000000000000,83.2000000000000,79.6000000000000,76])
    # Mot_trq_max_quarter = Mot_trq_max_quarter[np.newaxis, :]
    # Mot_trq_max_list = np.concatenate((np.fliplr(Mot_trq_max_quarter[:, 1:]), Mot_trq_max_quarter), axis = 1)    # 数组拼接函数，axis=1表示对应行上的拼接 
    # # motor minimum torque 
    # Mot_trq_min_list = - Mot_trq_max_list
    # Mot_trq_min_func = interp1d(Mot_spd_list, Mot_trq_min_list, kind = 'linear', fill_value = 'extrapolate')
    # Mot_trq_max_func = interp1d(Mot_spd_list, Mot_trq_max_list, kind = 'linear', fill_value = 'extrapolate')
    
    # # 以指定速度和加速度行驶的功率需求
    # Wheel_spd = car_spd/Wheel_R
    # ring_spd = Wheel_spd*Pgs_K
    # mg2_spd = ring_spd                                                               # 假设齿圈速度和MG2转速一致
    # F_roll = mass * G * C_roll * (T_factor if car_spd > 0 else 0)                    # 行驶方程式考虑滚动阻力，空气阻力，加速阻力，坡度为0
    # F_drag = 0.5 * density_air * area_frontal * C_d *(car_spd ** 2)
    # F_a = mass * car_a
    # T = Wheel_R * (F_roll + F_drag + F_a )                                                          # 轮端转矩
    # P_req = T * Wheel_spd   
    # #print("P_req: ", P_req)

    Eng_spd = 1500 * 2 * math.pi /60
    Wheel_spd = car_spd/Wheel_R
    ring_spd = Wheel_spd * Pgs_K
    mg2_spd = ring_spd
    mg1_spd = (1 + Pgs_R/Pgs_S)*Eng_spd - ring_spd * Pgs_R/Pgs_S 

    # print("Mg2-spd",ring_spd)
    # print("Eng-spd",Eng_spd)
    # print("mg1-spd",mg1_spd)

    return Eng_spd, mg1_spd, mg2_spd, Wheel_spd

# car_spd = 20
# car_a = 0.5
# A, B, C, D = calculate_speed(car_spd, car_a)
# print(A)
# print(B)
# print(C)

