# -*- coding: utf-8 -*-

import re
import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio
from Parameter import Param
param = Param()
    
def reward_traffic(location, Phase, remaining_time, speedlimit, car_spd, car_a, dis2inter):
    cur_phase = Phase
    next_phase = - (cur_phase - 1)
    brake_max = -2.5
    a_max = 2.5
    r_light = 0
    # # 首先针对到达路口的末端状态进行reward
    # if dis2inter == 0 and Phase == 1 and car_spd != 0 :
    #     # 以绿灯通过路口，鼓励
    #     r_light = 1
    # if dis2inter == 0 and Phase == 0 and car_spd != 0 :
    #     # 闯红灯，惩罚
    #     r_light = -1
    # if dis2inter == 0 and Phase == 0 and car_spd != 0 :
    #     # 红灯停住了，中立
    #     r_light = 0
    
    # 初始化
    segment_len = location.endpoint - location.startpoint
    d_cri = car_spd**2/2/(-brake_max)
    d_max = min(car_spd*remaining_time + 0.5*a_max*(remaining_time**2), speedlimit * remaining_time)
    d_miss = segment_len - dis2inter
    c1 = param.r_tra_ct1
    c2 = param.r_tra_ct2
    c3 = param.r_tra_ct3

    # 然后对靠近路口的状态进行reward
    if cur_phase == 1 :
        # 当前是绿灯，判断最大可行驶距离是否大于到路口距离
        if dis2inter > d_max:
            # 此绿灯过不去
            spd_flag = 2
        else:
            spd_flag = 1
    else:
        # 当前相位为红灯，判断临界制动距离是否小于到路口距离
        if d_cri <= dis2inter:
            # 能刹住车
            spd_flag = 0
        else:
            # 刹不住
            spd_flag = 3

    # 根据标志位求不同情况的reward
    if spd_flag == 1:
        r_tra = c1 + c2*d_miss
        if dis2inter > d_cri:
            spd_flag = 0
    elif spd_flag == 2:
        r_tra = c1 + c3*car_spd
        if dis2inter > d_cri:
            spd_flag = 0
    elif spd_flag == 3: 
        r_tra = c1 + c3*car_spd
    else:
        r_tra = c1 + c3*car_spd
        

    
    return r_tra

