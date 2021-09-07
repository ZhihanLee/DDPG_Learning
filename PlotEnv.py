# -*- coding: utf-8 -*-

import numpy as np
import os
from numpy.lib.function_base import disp
from scipy.interpolate import interp1d, interp2d, make_interp_spline
import matplotlib.pyplot as plt
import math
import scipy.io as scio
from Road import Road

def initEnv(t_list, displacement_list, i):
    time = []
    y = []
    y_inter = []
    # 去文本
    del t_list[0]
    del displacement_list[0]
    drawrunning(time, y, y_inter,t_list,displacement_list,i)



def drawpic(t, phase, y):
    if phase == 0:
        plt.scatter(t, y, s=2*2, marker = "s", c = '#FF0000')
    else:
        plt.scatter(t, y, s=2*2, marker = "s",c = '#008000')


def drawrunning(time, y, y_inter, t_list, displacement_list,i):
    episode_num = i
    # 类对象实例化
    RoadSegmentList = []
    # # # 能训练的数据（长路口有bug
    RoadSegmentList.append(Road(1, 60, 0, 525, 31, 49, -20))         # 嘉陵江东街
    RoadSegmentList.append(Road(2, 60, 525, 1480, 28, 51, -20))       # 白龙江东街
    RoadSegmentList.append(Road(3, 60, 1480, 2005, 30, 82, -20))      # 河西大街
    # # 参陈浩师兄多路口数据
    # RoadSegmentList.append(Road(1, 60, 0, 1200, 40, 60, -20))         # 嘉陵江东街
    # RoadSegmentList.append(Road(2, 80, 1200, 2200, 40, 65, -20))       # 白龙江东街
    # RoadSegmentList.append(Road(3, 80, 2200, 3700, 42, 65, -20))      # 河西大街
    # RoadSegmentList.append(Road(4, 80, 3700, 5100, 37, 55, -20))         # 嘉陵江东街
    # RoadSegmentList.append(Road(5, 80, 5100, 6400, 40, 65, -20))       # 白龙江东街
    # RoadSegmentList.append(Road(6, 80, 6400, 8000, 55, 67, -20))      # 河西大街
    # print("===============")
    # print(len(RoadSegmentList))
    # RoadSegmentList.append(Road(1, 60, 0, 326, 31, 49, -20))         # 嘉陵江东街
    # RoadSegmentList.append(Road(2, 65, 326, 679, 28, 51, -20))       # 白龙江东街
    # RoadSegmentList.append(Road(3, 60, 679, 1005, 30, 82, -20))      # 河西大街
    # RoadSegmentList.append(Road(4, 60, 1005, 1320, 38, 50, -30))      # 楠溪江东街
    # RoadSegmentList.append(Road(5, 60, 1320, 1606, 38, 50, 0))     # 富春江大街
    # RoadSegmentList.append(Road(6, 60, 1606, 1935, 27, 61, -30))     # 奥体大街
    # RoadSegmentList.append(Road(7, 60, 1935, 2232, 38, 50, 0))     # 新安江街
    # RoadSegmentList.append(Road(8, 60, 2232, 2500, 99, 1, 0))     # 终点路段

    for i in range(len(RoadSegmentList)):
        y_inter.append(RoadSegmentList[i].endpoint)

    for t in range(1,500):
        time.append(t)

        for j in range(len(RoadSegmentList)):
            Phasej, remainingj = RoadSegmentList[j - 1].SPaT(t)
            drawpic(t, Phasej, RoadSegmentList[j - 1].endpoint)


        # Phase1, remaining1 = RoadSegmentList[0].SPaT(t)
        # drawpic(t, Phase1, RoadSegmentList[0].endpoint)

        # Phase2, remaining2 = RoadSegmentList[1].SPaT(t)
        # drawpic(t, Phase2, RoadSegmentList[1].endpoint)

        # Phase3, remaining3 = RoadSegmentList[2].SPaT(t)
        # drawpic(t, Phase3, RoadSegmentList[2].endpoint)

        # Phase4, remaining1 = RoadSegmentList[3].SPaT(t)
        # drawpic(t, Phase4, RoadSegmentList[3].endpoint)

        # Phase5, remaining1 = RoadSegmentList[4].SPaT(t)
        # drawpic(t, Phase5, RoadSegmentList[4].endpoint)

        # Phase6, remaining1 = RoadSegmentList[5].SPaT(t)
        # drawpic(t, Phase6, RoadSegmentList[5].endpoint)

        # Phase7, remaining1 = RoadSegmentList[6].SPaT(t)
        # drawpic(t, Phase7, RoadSegmentList[6].endpoint)

        # Phase8, remaining1 = RoadSegmentList[7].SPaT(t)
        # drawpic(t, Phase8, RoadSegmentList[7].endpoint)

    # 用循环画相位图
    # 开始画车辆运动轨迹
    print("相位图画完了，下面画运动轨迹")
    result_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'Result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # plt.scatter(t_list, displacement_list, s=0.5*0.5, marker = "s", c = 'b')
    plt.plot(t_list, displacement_list, linewidth = '1', linestyle = '-', c = 'b')
    print("画完运动轨迹了")

    plt.xlabel("Time/s")
    plt.ylabel("Distance/m")
    picname = result_dir + '/' + str(episode_num) + ".png"
    plt.savefig(picname, dpi = 500, bbox_inches='tight')
    plt.close('all')


# # 没有车辆运动轨迹，画场景
# t_list = []
# for i in range(500):
#     t_list.append(i)

# i = 1
# displacement_list = []
# initEnv(t_list, displacement_list, i)
# plt.ylim(0,8500)
# picname = 'D:\\RL\\仿真图片\\' + '场景相位配时分布图' + '.png'
# plt.savefig(picname, dpi = 500, bbox_inches = 'tight')
# plt.close()

    






