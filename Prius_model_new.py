# -*- coding: utf-8 -*-
"""
the Model of Prius

now it with dynamic
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio

class Prius_model():
    def __init__(self):
        # paramsmeters of car
        self.Wheel_R = 0.287
        self.mass = 1449
        self.C_roll  = 0.013
        self.density_air = 1.2
        self.area_frontal = 2.52
        self.G = 9.81
        self.C_d = 0.26
        # the factor of F_roll
        self.T_factor = 0.015
        # 发动机怠速转速 100rad/s -> 955 rpm
        self.idling_speed = 100
       
        # paramsmeters of transmission system
        # number of teeth
        self.Pgs_R = 2.6                        # 齿圈齿数
        self.Pgs_S = 1                          # 太阳轮齿数
        # speed ratio from ring gear to wheel  从齿圈到车轮的传动比(主减速器传动比)
        self.Pgs_K = 3.93  
        
        # Inertia
        self.I_e = 0.18                         # 数据参考matlab 2010Prius
        self.I_r = 0
        self.I_c = 0
        self.I_s = 0
        self.I_v = self.mass * (self.Wheel_R**2)/(self.Pgs_K**2)                 # 参考matlab 2010 Prius
        # print("self.I_v = ",self.I_v)
        self.g_inertia = 0.0000226
        self.m_inertia = 0.015
        self.I_mg1 = self.g_inertia
        self.I_mg2 = self.m_inertia
        # max speed of ICE/MG1/MG2
        rpm2rad = math.pi/30
        self.Eng_spd_max = 4000 * rpm2rad
        self.Gen_spd_max = 10000 * rpm2rad
        self.Gen_spd_min = -10000 * rpm2rad
        self.Mot_spd_max = 6000 * rpm2rad


        # The optimal curve of engine   
        self.Eng_pwr_opt_list = np.arange(0, 57, 1) * 1000                                                                                             # 发动机功率范围
        #T_list = [0.0, 11.93662073189215, 23.8732414637843, 35.80986219567645, 47.7464829275686, 59.68310365946075, 71.6197243913529, 78.64126599834829, 84.88263631567752, 76.73541899073525, 82.3215222889114, 82.71044286665428, 84.25849928394459, 89.30996806595566, 83.03736161316279, 87.87696244337779, 88.31719385446216, 92.7645954021333, 98.22133630814113, 98.6068669156308, 97.94150344116636, 99.768770296412, 101.98277906859313, 104.09185851507847, 96.70173757482249, 105.16846459816874, 101.75479968170357, 103.54658948147409, 105.683914780389, 107.75470855248945, 109.34309067382122, 110.04765581818786, 110.7164821508837, 112.14476417151343, 113.92143294998826, 116.05047933784036, 117.73105379400477, 119.36620731892151, 120.95775674984046, 121.70672118791997, 121.64709026132127, 121.96920872463008, 121.90591385762197, 121.84562408815725, 121.7881303659721, 121.73324259153468, 121.68078751624131, 121.96112486933283, 121.58255599593066, 121.85300330473238, 121.80225236624644, 121.75353146529994, 121.70672118791995, 121.36995660245255, 121.90591385762195, 121.85877313300573, 121.81335052135952]
        self.W_list = [91.106186954104, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 83.77580409572782, 89.0117918517108, 94.24777960769379, 117.28612573401894, 121.47491593880532, 132.9940890019679, 142.4188669627373, 145.56045961632708, 168.59880574265222, 170.69320084504542, 181.1651763570114, 183.25957145940458, 183.25957145940458, 192.684349420174, 204.20352248333657, 210.48670779051614, 215.72269554649912, 220.95868330248211, 248.18581963359367, 237.71384412162766, 255.51620249196986, 260.7521902479528, 264.9409804527392, 269.1297706575256, 274.3657584135086, 281.69614127188476, 289.026524130261, 294.262511886244, 298.4513020910303, 301.59289474462014, 305.78168494940655, 309.9704751541929, 314.1592653589793, 320.4424506661589, 328.8200310757317, 336.15041393410786, 344.52799434368063, 352.90557475325346, 361.2831551628262, 369.660735572399, 378.0383159819718, 385.368698840348, 394.7934768011173, 402.12385965949346, 410.5014400690663, 418.87902047863906, 427.2566008882119, 436.6813788489813, 442.96456415616086, 451.3421445657336, 459.7197249753064]
        self.Eng_pwr_func = interp1d(self.Eng_pwr_opt_list, self.W_list)                                                                               # 一维插值  变成函数，得到功率和W曲线 W是角速度
        Eng_spd_list = np.arange(0, 4501, 125) * (2 * math.pi) / 60
        Eng_spd_list = Eng_spd_list[np.newaxis, :]
        Eng_trq_list = np.arange(0, 111, 5) * (121 / 110)
        Eng_trq_list = Eng_trq_list[np.newaxis, :]
        # BSFC的基本单位是g/kWh
        # 当单位是g/J时，换算方式是 g/kwh = g/j * 3600 * 1000
        data_path = 'Eng_bsfc_map.mat'
        data = scio.loadmat(data_path)
        Eng_bsfc_map = data['Eng_bsfc_map']     
        self.Eng_trq_maxP = [-4.1757e-009, 6.2173e-006, -3.4870e-003, 9.1743e-001, 2.0158e+001]                                                                   #多项式系数数组
        Eng_fuel_map = Eng_bsfc_map * (Eng_spd_list.T * Eng_trq_list) / 3600 / 1000   # 此处单位是g/J                        
        
        # fuel consumption (g)
        self.Eng_fuel_func = interp2d(Eng_trq_list, Eng_spd_list, Eng_fuel_map)                                                                                   #变成一个函数 可以用转矩和转速作为输入参数输出油耗
        
        # MG 2 
        # motor speed list (rad/s)
        Mot_spd_list = np.arange(-6000, 6001, 200) * (2 * math.pi) / 60        
        # motor torque list (Nm)
        Mot_trq_list = np.arange(-400, 401, 10)
        
        # motor efficiency map
        data_path1 = 'Mot_eta_quarter.mat'
        data1 = scio.loadmat(data_path1)
        Mot_eta_quarter = data1['Mot_eta_quarter']
        Mot_eta_alltrqs = np.concatenate(([np.fliplr(Mot_eta_quarter[:, 1:]), Mot_eta_quarter]), axis = 1)
        Mot_eta_map = np.concatenate(([np.flipud(Mot_eta_alltrqs[1:, :]), Mot_eta_alltrqs]))
        # motor efficiency
        self.Mot_eta_map_func = interp2d(Mot_trq_list, Mot_spd_list, Mot_eta_map)                                                                                 #变成一个函数 可以用转矩和转速作为输入参数输出油耗             
        
        #  motor maximum torque
        Mot_trq_max_quarter = np.array([400,400,400,400,400,400,400,347.200000000000,297.800000000000,269.400000000000,241,221.800000000000,202.600000000000,186.400000000000,173.200000000000,160,148,136,126.200000000000,118.600000000000,111,105.800000000000,100.600000000000,96.2000000000000,92.6000000000000,89,87.4000000000000,85.8000000000000,83.2000000000000,79.6000000000000,76])
        Mot_trq_max_quarter = Mot_trq_max_quarter[np.newaxis, :]
        Mot_trq_max_list = np.concatenate((np.fliplr(Mot_trq_max_quarter[:, 1:]), Mot_trq_max_quarter), axis = 1)    # 数组拼接函数，axis=1表示对应行上的拼接 
        # motor minimum torque 
        Mot_trq_min_list = - Mot_trq_max_list
        self.Mot_trq_min_func = interp1d(Mot_spd_list, Mot_trq_min_list, kind = 'linear', fill_value = 'extrapolate')
        self.Mot_trq_max_func = interp1d(Mot_spd_list, Mot_trq_max_list, kind = 'linear', fill_value = 'extrapolate')

        # Generator (MG 1)          
        # generator speed list (rad/s)
        Gen_spd_list = np.arange(-10e3, 11e3, 1e3) * (2 * math.pi) / 60
        Gen_trq_list = np.arange(-75, 76, 5) 
        
        # motor efficiency map
        Gen_eta_quarter = np.array([[0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000,0.570000000000000],[0.570000000000000,0.701190476190476,0.832380952380952,0.845500000000000,0.845500000000000,0.845500000000000,0.845500000000000,0.841181818181818,0.832545454545455,0.825204545454546,0.820886363636364,0.816136363636364,0.807500000000000,0.798863636363636,0.794113636363636,0.789795454545455],[0.570000000000000,0.710238095238095,0.850476190476190,0.872272727272727,0.880909090909091,0.883500000000000,0.883500000000000,0.879181818181818,0.870545454545455,0.864500000000000,0.864500000000000,0.863636363636364,0.855000000000000,0.846363636363636,0.841613636363636,0.837295454545455],[0.570000000000000,0.710238095238095,0.850476190476190,0.872272727272727,0.880909090909091,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.875727272727273,0.867090909090909],[0.570000000000000,0.710238095238095,0.850476190476190,0.876159090909091,0.889113636363636,0.896022727272727,0.900340909090909,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.894727272727273,0.886090909090909],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.902500000000000,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.714761904761905,0.859523809523810,0.885659090909091,0.898613636363636,0.902500000000000,0.902500000000000,0.898181818181818,0.889545454545454,0.886090909090909,0.894727272727273,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000],[0.570000000000000,0.710238095238095,0.850476190476190,0.872272727272727,0.880909090909091,0.883500000000000,0.883500000000000,0.883500000000000,0.883500000000000,0.886090909090909,0.894727272727273,0.901636363636364,0.893000000000000,0.884363636363636,0.883500000000000,0.883500000000000]])
        Gen_eta_alltrqs = np.concatenate((Gen_eta_quarter[:, 1:], Gen_eta_quarter), axis = 1)    
        Gen_eta_map = np.concatenate(([np.flipud(Gen_eta_alltrqs[1:, :]), Gen_eta_alltrqs]))
        # efficiency of the electric generator
        self.Gen_eta_map_func = interp2d(Gen_trq_list, Gen_spd_list, Gen_eta_map)                                                                                 #变成一个函数 可以用转矩和转速作为输入参数输出油耗          

        # generator maxmium torque
        Gen_trq_max_half = np.array([76.7000000000000,76.7000000000000,70.6160000000000,46.0160000000000,34.7320000000000,26.6400000000000,21.2000000000000,18.1160000000000,16.2800000000000,13.4000000000000,0])
        Gen_trq_max_half = Gen_trq_max_half[np.newaxis, :]
        Gen_trq_max_list = np.concatenate((np.fliplr(Gen_trq_max_half[:, 1:]), Gen_trq_max_half), axis = 1)
        # generator minimum torque
        Gen_trq_min_list = -Gen_trq_max_list
        self.Gen_trq_min_func = interp1d(Gen_spd_list, Gen_trq_min_list, kind = 'linear', fill_value = 'extrapolate')
        self.Gen_trq_max_func = interp1d(Gen_spd_list, Gen_trq_max_list, kind = 'linear', fill_value = 'extrapolate')

        # Battery
        # published capacity of one battery cell
        Batt_Q_cell = 6.5     
        # coulombs, battery package capacity
        self.Batt_Q = Batt_Q_cell * 3600     
        # resistance and OCV list                     OCV: Open Circuit Voltage
        Batt_rint_dis_list = [0.7,0.619244814,0.443380117,0.396994948,0.370210379,0.359869599,0.364414573,0.357095093,0.363394618,0.386654377,0.4] # ohm
        Batt_rint_chg_list = [0.7,0.623009741,0.477267027,0.404193372,0.37640518,0.391748667,0.365290105,0.375071555,0.382795632,0.371566564,0.36] # ohm
        Batt_vol_list  = [202,209.3825073,213.471405,216.2673035,218.9015961,220.4855042,221.616806,222.360199,224.2510986,227.8065948,237.293396] # V
        # resistance and OCV
        SOC_list = np.arange(0, 1.01, 0.1) 
        self.Batt_vol_func = interp1d(SOC_list, Batt_vol_list, kind = 'linear', fill_value = 'extrapolate')
        self.Batt_rint_dis_list_func = interp1d(SOC_list, Batt_rint_dis_list, kind = 'linear', fill_value = 'extrapolate')                                       # SOC 和 充电放电电阻 函数
        self.Batt_rint_chg_list_func = interp1d(SOC_list, Batt_rint_chg_list, kind = 'linear', fill_value = 'extrapolate')  

        #Battery current limitations
        self.Batt_I_max_dis = 196
        self.Batt_I_max_chg = 120 
        
        
    def run(self, car_spd, omega_list, T_list, SOC, STEP_SIZE):                                                  # T_list = [T_e, T_mg1, T_mg2]    omega_list = [Eng_spd, Gen_spd, Mot_spd]
        # Wheel speed (rad/s)          
        Wheel_spd = car_spd / self.Wheel_R                                                                   # 轮速用的是角速度

        # Wheel torque (Nm) 
        F_roll = self.mass * self.G * self.C_roll * (self.T_factor if car_spd > 0 else 0)                    # 行驶方程式考虑滚动阻力，空气阻力，加速阻力，坡度为0
        F_drag = 0.5 * self.density_air * self.area_frontal * self.C_d *(car_spd ** 2)
        # F_a = self.mass * car_a
        F_t = F_roll + F_drag                                                                          # 行驶方程式定义上的驱动力（阻力+惯性阻力）
        T = self.Wheel_R * ( F_roll + F_drag )                                                         # 轮端转矩
        
        # 三个转矩动作提取
        Eng_trq = T_list[0]
        Gen_trq = T_list[1]  # mg1
        Mot_trq = T_list[2]  # mg2

        #################转速异常排除###################
        # 当本步长初始的发动机转速低于怠速转速，不采用智能体输出值，改用负常阻力矩，重新计算角加速度更新发动机角速度
        if (omega_list[0] < self.idling_speed) :
            Eng_trq = -20
            T_list[0] = Eng_trq

        # 状态方程
        # D矩阵
        D = np.matrix([[self.I_c + self.I_e, 0, 0, self.Pgs_R + self.Pgs_S],[0, self.I_s + self.I_mg1, 0, -self.Pgs_S],[0, 0, self.I_v + self.I_mg2, - self.Pgs_R],[self.Pgs_R + self.Pgs_S, - self.Pgs_S, - self.Pgs_R, 0]])
        T_list.append(0)
        F_mat = [0, 0, F_t/self.Pgs_K, 0]

        # 转为矩阵类型并转置
        T_list = np.matrix(T_list)
        F_mat = np.matrix(F_mat)
        T_list = T_list.T
        F_mat = F_mat.T

        # 求角加速度
        alpha_list = D.I * (T_list - F_mat)                                                                  # alpha_list = [alhpa_e, alpha_mg1, alpha_mg2, F_pgs] = [行星架角加速度， 太阳轮角加速度， 齿圈角加速度, 轮系内力]
        alpha_list = np.array(alpha_list)                                                                    # 数据类型要变回列表

        # 更新车辆加速度
        car_a = self.Wheel_R * alpha_list[2]/self.Pgs_K
        if (car_a > 2.5) or (car_a < -2.5):
            inf_acc = 1

        # 车速更新 步长时间更新
        car_spd_ini = car_spd
        if (2*car_a*STEP_SIZE + car_spd_ini**2) > 0 :                                                                  # 先存当前步长初速度
            car_spd = math.sqrt(2*car_a*STEP_SIZE + car_spd_ini**2)
        else:
            car_spd = 0
        t = (car_a != 0)*(car_spd - car_spd_ini)/car_a # + (car_a == 0)*(STEP_SIZE/car_spd_ini)

        # 求离散步长末角速度，更新omega_list
        Eng_spd_ini = omega_list[0]
        Gen_spd_ini = omega_list[1]
        Mot_spd_ini = omega_list[2]
        omega_list[0] = omega_list[0] + alpha_list[0]*t
        omega_list[1] = omega_list[1] + alpha_list[1]*t
        omega_list[2] = omega_list[2] + alpha_list[2]*t
        omega_list[0] = float(omega_list[0])
        omega_list[1] = float(omega_list[1])
        omega_list[2] = float(omega_list[2])
        # # 用更新后的电机角速度重新算车速
        # # 于是下面的电机转速越界需要重新设置
        # car_spd = self.Wheel_R*omega_list[2]/self.Pgs_K

        if t == 0 :
            # 如果t=0，说明始末速度都为0，车停了
            stop_flag = 1
        else:
            stop_flag = 0

        # 校验更新后的角速度不能超过角速度最大值
        # 初始化标志位
        inf_mot = 0
        inf_gen = 0
        inf_eng = 0
        inf_acc = 0
        # 这里因为使用if修正，如果不止一个转速越界，则必须考虑先后次序，否则修正后的转速进入下一个if条件内可能又被修改
        # 首先以电动机作为第一修正对象，他的速度受车速约束；因此首先修正电动机符合车速，然后发电机因为是主动控制对象，保持其状态不变
        # 发动机进而依据行星系特性更新转速。这样更新完发动机后再检查发动机转速是否不合适
        if (abs(omega_list[2]) > self.Mot_spd_max) :
            # omega_list[2] = self.Pgs_K * car_spd/self.Wheel_R
            omega_list[2] = self.Mot_spd_max # 直接约束电机转速等于最大转速
            car_spd = self.Wheel_R*omega_list[2]/self.Pgs_K # 车速重新修正
            # 发电机的转速因为是主动调节的，所以这里就遵循上文更新策略，不修正
            omega_list[0] = (omega_list[1] + omega_list[2] * self.Pgs_R/self.Pgs_S) / (1 + self.Pgs_R/self.Pgs_S) # 最后一项服从行星系
            inf_mot = 1
            # print("Mot_spd error")

        # 如果发动机转速超过最大转速，发动机按给定阈值来 ；电动机依然维持车速约束，利用行星系求解发电机的转速
        if (omega_list[0] > self.Eng_spd_max) :
            omega_list[0] = self.Eng_spd_max # 发动机按最大转速给
            omega_list[2] = self.Pgs_K * car_spd/self.Wheel_R # 电机速度按齿圈速度-按车速给
            omega_list[1] = (1 + self.Pgs_R/self.Pgs_S)*omega_list[0] - omega_list[2] * self.Pgs_R/self.Pgs_S # 最后一项服从行星系
            inf_eng = 1
            # print("Eng-spd error for too high")
        if (omega_list[0] < 0) :
            # 如果更新的角速度小于0，约束到0
            omega_list[0] = 0
            omega_list[2] = self.Pgs_K * car_spd/self.Wheel_R # 电机速度按齿圈速度-按车速给
            omega_list[1] = (1 + self.Pgs_R/self.Pgs_S)*omega_list[0] - omega_list[2] * self.Pgs_R/self.Pgs_S # 最后一项服从行星系
            inf_eng = 1
            # print("Eng-spd error for too low")

        # 如果发电机不幸越界，重新约束发电机
        if (omega_list[1] > self.Gen_spd_max) :
            omega_list[1] = self.Gen_spd_max # 发电机按最大转速给
            omega_list[2] = self.Pgs_K * car_spd/self.Wheel_R # 电机速度依然服从车速
            omega_list[0] = (omega_list[1] + omega_list[2] * self.Pgs_R/self.Pgs_S) / (1 + self.Pgs_R/self.Pgs_S) # 最后一项服从行星系
            inf_gen = 1
            # print("Gen-spd error")
        if (omega_list[1] < self.Gen_spd_min) :
            omega_list[1] = self.Gen_spd_min # 发电机按最小转速给
            omega_list[2] = self.Pgs_K * car_spd/self.Wheel_R # 电机速度依然服从车速
            omega_list[0] = (omega_list[1] + omega_list[2] * self.Pgs_R/self.Pgs_S) / (1 + self.Pgs_R/self.Pgs_S) # 最后一项服从行星系
            inf_gen = 1

        # 求步长内角速度均值
        Eng_spd = (Eng_spd_ini + omega_list[0])/2
        Eng_spd = np.array(Eng_spd,dtype='float64')
        Gen_spd = (Gen_spd_ini + omega_list[1])/2
        Gen_spd = np.array(Gen_spd,dtype='float64')
        Mot_spd = (Mot_spd_ini + omega_list[2])/2
        Mot_spd = np.array(Mot_spd,dtype='float64')
        # 用步长内定转矩和角速度均值和通行时间计算能耗
        # 发动机的
        Eng_fuel_mdot = self.Eng_fuel_func(Eng_trq, Eng_spd) # 单位g 实际可以理解为在这个离散步长内，发动机保持在了能耗为eng_ful_mdot克，转速转矩为xx的工况点上
        Eng_pwr = Eng_fuel_mdot * 42600                      # 42600为热值，一般单位kj/kg，Eng_pwr表示当前发动机功率（瓦特） = 当前发动机工作单位时间(1s)可以释放的能量（焦耳） 
        Eng_pwr = np.array(Eng_pwr, dtype='float64')         # 单位瓦特
        # Mot的
        Mot_eta = (Mot_spd == 0) + (Mot_spd != 0) * self.Mot_eta_map_func(Mot_trq, Mot_spd * np.ones(1)) #need to edit 
        Mot_eta[np.isnan(Mot_eta)] = 1
        Mot_pwr = (Mot_trq * Mot_spd <= 0) * Mot_spd * Mot_trq * Mot_eta + (Mot_trq * Mot_spd > 0) * Mot_spd * Mot_trq / Mot_eta # 分类讨论 电机功率输出为负，说明电池在被充电，乘以效率； 电机功率输出为正，则换算到电池上功率要除以效率
        Mot_pwr = np.array(Mot_pwr, dtype='float64')
        # Gen的
        Gen_eta = (Gen_spd == 0) + (Gen_spd != 0) * self.Gen_eta_map_func(Gen_trq, Gen_spd)
        Gen_eta[np.isnan(Gen_eta)] = 1
        Gen_pwr = (Gen_trq * Gen_spd <= 0) * Gen_spd * Gen_trq * Gen_eta + (Gen_trq * Gen_spd > 0) * Gen_spd * Gen_trq / Gen_eta # 他这里约定是功率为负，则说明在发电，冲到电池的能量要乘以效率；反之在用电，消耗电池的能量要除以效率
        Gen_pwr = np.array(Gen_pwr, dtype='float64')

        P_out = (Eng_trq * omega_list[0] *30/math.pi + Gen_trq * omega_list[1]*30/math.pi + Mot_trq * omega_list[2]*30/math.pi)/9550  # kW

        SOC_new = 0.65
        delta_SOC = SOC_new - SOC
        Batt_pwr = Mot_pwr + Gen_pwr                                               
        
        # Cost
        #I = (inf_batt + inf_eng + inf_mot + inf_gen != 0)                                                                              # 我也不知掉他是干啥的，先注释了
        # Calculate cost matrix (fuel mass flow)
        # I = (inf_eng == 1)*1 + (inf_gen == 1)*2 + (inf_mot == 1)*4 + (inf_acc == 1)*9 + (inf_soc == 1)*63
        I = (inf_eng == 1)*1 + (inf_gen == 1)*2 + (inf_mot == 1)*4 + (inf_acc == 1)*9 + (stop_flag == 1)*20
        cost = (Eng_pwr / 42600)
        
        out = {}
        # out['P_req'] = P_req
        # out['P_out'] = P_out
        # out['Eng_spd'] = Eng_spd
        # out['Eng_trq'] = Eng_trq
        # out['Eng_pwr'] = Eng_pwr 
        # #out['Eng_pwr_opt'] = Eng_pwr_opt
        # out['Mot_spd'] = Mot_spd
        # out['Mot_trq'] = Mot_trq
        # out['Mot_pwr'] = Mot_pwr  
        # out['Gen_spd'] = Gen_spd
        # out['Gen_trq'] = Gen_trq
        # out['Gen_pwr'] = Gen_pwr
        out['SOC'] = SOC_new 
        out['delta_SOC'] = delta_SOC   
        # out['Batt_vol'] = Batt_vol       
        out['Batt_pwr'] = Batt_pwr
        # out['inf_batt'] = inf_batt
        # out['inf_batt_one'] = inf_batt_one
        # out['T'] = T
        # out['Mot_eta'] = Mot_eta
        # out['Gen_eta'] = Gen_eta
        
        return  out, cost, I, car_spd, car_a, omega_list, alpha_list, t

#Prius = Prius_model()
#out, cost, I = Prius.run(20, 1, 30000, 0.8)
