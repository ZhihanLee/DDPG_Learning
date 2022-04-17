# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio

class Road(object):
    # 规定路的编码，速度上限，全局坐标下的起点、终点，绿灯时间，红灯时间
    def __init__(self, id, maxspeed, startpoint, endpoint, GreenTiming, RedTiming, delay):
        self.segment_id = id
        self.maxspeed = maxspeed/3.6
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.GreenTiming = GreenTiming
        self.RedTiming = RedTiming
        self.delay = delay

    def SPaT(self, t):
        Green = self.GreenTiming
        Red = self.RedTiming
        time = t % (Green + Red)
        if time <= Red:
            # 现在是红灯相位
            Phase = 0
            remaining_time = Red - time
        else:
            # 现在是绿灯相位
            Phase = 1
            remaining_time = Red + Green - time

        return Phase, remaining_time


        
## 局部测试：
# displacement = 999
# #t = 300
# Com_range = 300 # V2I通信距离

# RoadSegmentList = []
# RoadSegmentList.append(Road(1, 60, 0, 598, 30, 30))
# RoadSegmentList.append(Road(2, 65, 598, 831, 80, 40))
# RoadSegmentList.append(Road(3, 60, 831, 1196, 40, 80))
# RoadSegmentList.append(Road(4, 60, 1196, 1513, 50,50))
# RoadSegmentList.append(Road(5, 60, 1513, 1996, 35, 35))
# RoadSegmentList.append(Road(6, 60, 1996, 2332, 40, 40))

# for t in range(1,500):
#     for i in range(len(RoadSegmentList)):
#         if (displacement <= RoadSegmentList[i].endpoint) & (displacement > RoadSegmentList[i].startpoint):
#             location = RoadSegmentList[i]
#             break
#         i = i + 1
#     speedlimt = location.maxspeed

#     print("M IN ROAD ", i+1)

#     if (displacement + Com_range) >= location.endpoint:
#         Phase, remaining_time = location.SPaT(t)
#         dis2inter = location.endpoint - displacement
#     else:
#         Phase = 1           
#         remaining_time = 999
#     print(Phase)
#     print(remaining_time)






