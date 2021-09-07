from sys import api_version
import gym
import os
import numpy as np
import math

from Prius_model_new import Prius_model
from Fuel_Consumption import get_fuel
from State_Machine import reward_traffic
import PlotEnv as ple
from calculateBestOmega import calculate_speed
from Road import Road
import csv
from Parameter import Param
param = Param()

# class Init_Param():
# 	def __init__(self) -> None:
# 		pass

#####################  PATH   ###########################
reward_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'reward')
if not os.path.exists(reward_dir):
    os.makedirs(reward_dir)
data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
SOC_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'soc')
if not os.path.exists(SOC_dir):
    os.makedirs(SOC_dir)
Test_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'Testing')
if not os.path.exists(Test_dir):
    os.makedirs(Test_dir)

class Env():
	def __init__(self):
		# 道路类对象实例化
		self.RoadSegmentList = []
		# # 能训练的数据
		self.RoadSegmentList.append(Road(1, 60, 0, 525, 31, 49, -20))         # 嘉陵江东街
		self.RoadSegmentList.append(Road(2, 60, 525, 1480, 28, 51, -20))       # 白龙江东街
		self.RoadSegmentList.append(Road(3, 60, 1480, 2005, 30, 82, -20))      # 河西大街
		# # 参陈浩师兄多路口数据
		# self.RoadSegmentList.append(Road(1, 60, 0, 1200, 40, 60, -20))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(2, 80, 1200, 2200, 40, 65, -20))       # 白龙江东街
		# self.RoadSegmentList.append(Road(3, 80, 2200, 3700, 42, 65, -20))      # 河西大街
		# self.RoadSegmentList.append(Road(4, 80, 3700, 5100, 37, 55, -20))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(5, 80, 5100, 6400, 40, 65, -20))       # 白龙江东街
		# self.RoadSegmentList.append(Road(6, 80, 6400, 8000, 55, 67, -20))      # 河西大街
		# 庐山路实际数据
		# self.RoadSegmentList.append(Road(1, 60, 0, 326, 31, 49, -20))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(2, 65, 326, 679, 28, 51, -20))       # 白龙江东街
		# self.RoadSegmentList.append(Road(3, 60, 679, 1005, 30, 82, -20))      # 河西大街
		# self.RoadSegmentList.append(Road(4, 60, 1005, 1320, 38, 50, -30))      # 楠溪江东街
		# self.RoadSegmentList.append(Road(5, 60, 1320, 1606, 38, 50, 0))     # 富春江大街
		# RoadSegmentList.append(Road(6, 60, 1606, 1935, 27, 61, -30))     # 奥体大街
		# RoadSegmentList.append(Road(7, 60, 1935, 2232, 38, 50, 0))     # 新安江街
		# # RoadSegmentList.append(Road(8, 60, 2232, 2500, 99, 1, 0))     # 终点路段  

		# 环境量初始化
		self.s_dim = 8
		self.a_dim = 3
		self.t_total = 0
		self.STEP_SIZE = param.step_size
		self.displacement = 0
		self.dis2inter = self.RoadSegmentList[0].endpoint                   # 初始化成刚好在通信范围外，是否合适有待验证
		self.dis2termin = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint
		self.travellength = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint
		self.location = self.RoadSegmentList[0].segment_id                      # 目前车辆所在路段编号
		self.speedlimit = self.RoadSegmentList[0].maxspeed                  # 道路限速初始化
		self.signal_flag = 0                   # 能否收到相位配时的标志位

		# 重要状态初始化
		self.SOC_min = 0.4
		self.SOC_max = 0.8
		self.SOC_origin = 0.65
		self.isdone = 0

		# 需要导出的数据存储列表
		self.car_spd_list = []
		self.car_spd_list.append("Speed")
		self.car_a_list = []
		self.car_a_list.append("Acce")
		self.SOC_data = []
		self.SOC_data.append("SOC")
		self.Eng_spd_list = []
		self.Eng_spd_list.append("Engine Speed")
		self.Eng_trq_list = []
		self.Eng_trq_list.append("Engine Torque")
		self.Eng_pwr_list = []
		self.Eng_pwr_opt_list = []
		self.Gen_spd_list = []
		self.Gen_spd_list.append("Generator Speed")
		self.Gen_trq_list = []
		self.Gen_trq_list.append("Generator Torque")
		self.Gen_pwr_list = []
		self.Mot_spd_list = []
		self.Mot_spd_list.append("Motor Speed")
		self.Mot_trq_list = []
		self.Mot_trq_list.append("Motor Torque")
		self.Eng_alp_list = []
		self.Eng_alp_list.append("Eng Alp")
		self.Gen_alp_list = []
		self.Gen_alp_list.append("Gen Alp")
		self.Mot_alp_list = []
		self.Mot_alp_list.append("Mot Alp")

		self.Phase_list = []
		self.Phase_list.append("Phase")
		self.remaining_time_list = []  
		self.remaining_time_list.append("RemainTime")
		self.signal_flag_list = []
		self.signal_flag_list.append("SignalFlag")
		self.dis2inter_list = []
		self.dis2inter_list.append("dis2inter")

		self.displacement_list = []
		self.displacement_list.append("s")
		self.t_list = [] 
		self.t_list.append("time")

		self.action_list = []   
		self.action_list.append("action")  

		self.eq_fuel_cost_list = []
		self.eq_fuel_cost_list.append("eq-fuel-cost")
		self.Reward_list_all = []
		self.Reward_list_all.append("AllReward")

		self.I_list = []
		self.I_list.append("illegal")

		# 记录每个episode里每一step的单项reward,需要在episode开始时清空
		self.r_fuel_list = []
		self.r_fuel_list.append("r_fuel")
		self.r_soc_list = []
		self.r_soc_list.append("r_soc")
		self.r_tra_list = []
		self.r_tra_list.append("r_tra")
		self.r_spd_list = []
		self.r_spd_list.append("r_spd")
		self.r_illegal_list = []
		self.r_illegal_list.append("r_ill")
		self.r_success_list = []
		self.r_success_list.append("r_suc")

		# 用来画单项reward变化的数组
		self.mean_fuel_list = []
		self.mean_soc_list = []
		self.mean_tra_list = []
		self.mean_spd_list = []		
		self.mean_ill_list = []
		self.mean_suc_list = []	

		# DRL-EMS源码中的列表，我不一定用到的
		self.cost_Engine_list = []
		self.cost_all_list = []
		self.cost_Engine_100Km_list = []
		self.mean_reward_list = []
		self.list_even = []
		self.list_odd = []
		self.mean_discrepancy_list = []
		self.SOC_final_list = []

	
	def reset(self): # 初始化
		car_spd = 20                      # KM/h
		car_spd = car_spd/3.6
		car_a = 0
		SOC = self.SOC_origin
		self.t_total = 0
		self.displacement = 0
		# 计算初始状态下合理的转速分配
		Eng_spd, Gen_spd, Mot_spd, Wheel_spd = calculate_speed(car_spd)
		omega_list = [Eng_spd, Gen_spd, Mot_spd]
		s = np.zeros(self.s_dim)
		# s[0] = car_spd
		# s[1] = car_a
		# s[2] = SOC
		# s[3] = self.RoadSegmentList[0].maxspeed
		# s[4] = self.RoadSegmentList[0].endpoint # dis2inter
		# s[5] = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint # dis2termin
		# s[6], s[7] = self.RoadSegmentList[0].SPaT(0)
		# normalized input
		s[0] = car_spd/self.RoadSegmentList[0].maxspeed
		s[1] = np.clip(abs(car_a)/2.5, 0., 1.)
		s[2] = SOC
		s[3] = self.RoadSegmentList[0].endpoint / (self.RoadSegmentList[0].endpoint) # dis2inter/路段长
		s[4] = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint / self.travellength # dis2termin/总长
		s[5], timing = self.RoadSegmentList[0].SPaT(0)
		s[6] = (s[5] == 1)*timing/self.RoadSegmentList[0].GreenTiming + (s[5] == 0)*timing/self.RoadSegmentList[0].RedTiming
		s[7] = 0 # reward
		self.isdone = 0
		# 清空数据记录列表
		self.car_spd_list = []
		self.car_spd_list.append("Speed")
		self.car_a_list = []
		self.car_a_list.append("Acce")
		self.SOC_data = []
		self.SOC_data.append("SOC")
		self.Eng_spd_list = []
		self.Eng_spd_list.append("Engine Speed")
		self.Eng_trq_list = []
		self.Eng_trq_list.append("Engine Torque")
		self.Eng_pwr_list = []
		self.Eng_pwr_opt_list = []
		self.Gen_spd_list = []
		self.Gen_spd_list.append("Generator Speed")
		self.Gen_trq_list = []
		self.Gen_trq_list.append("Generator Torque")
		self.Gen_pwr_list = []
		self.Mot_spd_list = []
		self.Mot_spd_list.append("Motor Speed")
		self.Mot_trq_list = []
		self.Mot_trq_list.append("Motor Torque")
		self.Eng_alp_list = []
		self.Eng_alp_list.append("Eng Alp")
		self.Gen_alp_list = []
		self.Gen_alp_list.append("Gen Alp")
		self.Mot_alp_list = []
		self.Mot_alp_list.append("Mot Alp")

		self.Phase_list = []
		self.Phase_list.append("Phase")
		self.remaining_time_list = []  
		self.remaining_time_list.append("RemainTime")
		self.signal_flag_list = []
		self.signal_flag_list.append("SignalFlag")
		self.dis2inter_list = []
		self.dis2inter_list.append("dis2inter")

		self.displacement_list = []
		self.displacement_list.append("s")
		self.t_list = [] 
		self.t_list.append("time")

		self.action_list = []   
		self.action_list.append("action")  

		self.cost_Engine_list = []
		self.cost_Engine_list.append("fuel_cost")
		self.eq_fuel_cost_list = []
		self.eq_fuel_cost_list.append("eq-fuel-cost")
		self.Reward_list_all = []
		self.Reward_list_all.append("AllReward")

		self.I_list = []
		self.I_list.append("illegal")

		self.r_fuel_list = []
		self.r_fuel_list.append("r_fuel")
		self.r_soc_list = []
		self.r_soc_list.append("r_soc")
		self.r_tra_list = []
		self.r_tra_list.append("r_tra")
		self.r_spd_list = []
		self.r_spd_list.append("r_spd")
		self.r_illegal_list = []
		self.r_illegal_list.append("r_ill")
		self.r_success_list = []
		self.r_success_list.append("r_suc")

		# DRL-EMS源码中的列表，我不一定用到的
		# self.cost_Engine_list = [] # 这些都是以episode为单位记录的，不可以被reset方法在每个episode开始时重置
		# self.cost_all_list = []
		# self.cost_Engine_100Km_list = []
		# self.mean_reward_list = []
		# self.list_even = []
		# self.list_odd = []
		# self.mean_discrepancy_list = []
		# self.SOC_final_list = []

		return s, omega_list
	
	def step(self, T_list, s, omega_list): # 离散步长内的所有计算都发生在这里
		##### 确定上一步长末端的道路限速 #####
		for h in range(len(self.RoadSegmentList)):                                     # 定位车辆在哪段路上，返回该段路的最高时速
			# print("displacement = ",self.displacement)
			# print("h段终点：", self.RoadSegmentList[h].endpoint)
			if (self.displacement <= self.RoadSegmentList[h].endpoint) and (self.displacement > self.RoadSegmentList[h].startpoint):
				speedlimit_last = self.RoadSegmentList[h].maxspeed
				#print("目前车辆行驶在ROAD ", h)
				break
			if self.displacement == self.RoadSegmentList[h].startpoint:
				speedlimit_last = self.RoadSegmentList[h].maxspeed
				break
			h = h + 1

		##### 更新车辆运动学动力学参数 #####
		Prius = Prius_model()
		out, cost, I, car_spd, car_a, omega_list, alpha_list, t = Prius.run(s[0]*speedlimit_last, omega_list, T_list, s[2], self.STEP_SIZE)
		car_a = float(car_a)
		car_spd = float(car_spd)

		##### 获得油耗率 #####
		fuel_cost_rate = float(cost)

		##### 获得电耗 #####
		SOC_new = out['SOC']
		SOC_new = float(SOC_new)
		delta_soc = out['delta_SOC']
		bat_pwr = out['Batt_pwr']

		##### 更新位移 #####
		# 排除车速异常，如果出现需要修正一下时间和位移
		if (car_a < 0)and(car_spd == 0) :
			# 如果更新之后车速是0 加速度为负，则车要停到该步长的末端
			# 下一个步长如果转矩不变化，时间增量为0，所以上面的t_total不用修正，位移保持不变
			# 否则的话，位移可以继续更新
			# print("停车了，妈的")
			self.displacement = self.displacement
		else:
			self.displacement = self.displacement + self.STEP_SIZE
		
		self.dis2termin = self.travellength - self.displacement
		
		##### 车辆定位 #####
		for h in range(len(self.RoadSegmentList)):                                     # 定位车辆在哪段路上，返回该段路的最高时速
			# print("displacement = ",self.displacement)
			# print("h段终点：", self.RoadSegmentList[h].endpoint)
			if (self.displacement <= self.RoadSegmentList[h].endpoint) and (self.displacement > self.RoadSegmentList[h].startpoint):
				location = self.RoadSegmentList[h]
				#print("目前车辆行驶在ROAD ", h)
				break
			if self.displacement == self.RoadSegmentList[h].startpoint:
				location = self.RoadSegmentList[h]
				break
			h = h + 1
		speedlimit = location.maxspeed

		##### 更新时间 #####
		self.t_total = self.t_total + t
		self.t_total = float(self.t_total)

		##### 更新SPaT #####
		Phase, remaining_time = location.SPaT(self.t_total)
		# print("==============")
		# print("t_total",self.t_total)
		# print("remaining_time",remaining_time)
		self.dis2inter = location.endpoint - self.displacement
		#signal_flag_list.append(signal_flag)

		##### 计算reward #####
		# # 求r_fuel
		fuel_cost, eq_fuel_cost = get_fuel(fuel_cost_rate, bat_pwr, SOC_new, delta_soc, t) 
		fuel_cost = float(fuel_cost)
		# cf = param.r_fuel_cf
		# ct = param.r_fuel_ct
		# r_fuel = cf * (fuel_cost + ct)

		# # 求r_soc
		# cbat1 = param.r_soc_bat1
		# cbat2 = param.r_soc_bat2
		# r_SOC = (self.displacement < self.travellength) * (cbat1) * (max(SOC_new - self.SOC_max, 0) + max(self.SOC_min - SOC_new, 0)) +\
		# 		(self.displacement == self.travellength) * (cbat2) * max(SOC_new - self.SOC_origin, 0)
		# if SOC_new < self.SOC_min :
		# 	# 惩罚SOC过低，避免负数
		# 	r_SOC = param.illegal_soc_punish

		# # 求r_tra
		# r_tra = reward_traffic(location, Phase, remaining_time, speedlimit, car_spd, car_a, self.dis2inter)

		# # 求r_spd
		# cv1 = param.r_spd_cv1
		# cv2 = param.r_spd_cv2
		# r_spd = cv1*max((car_spd - speedlimit), 0) + (t>0) * cv2 * ((car_a - s[1])/t)**2 - cv1 * (car_spd > 10/3.6) * car_spd
		# if car_spd <= 10/3.6:
		# 	# 惩罚车速过低，避免停车
		# 	r_spd = param.illegal_spd_punish

		# # 日常奖励
		# r_moving = -s[3] - car_spd * param.r_spd_cv2 # cv2 = -0.015
		r_moving = 200*(s[4] - self.dis2termin/self.travellength)*param.r_spd_cv1 + ((car_spd - speedlimit)**2 + (car_spd - 10/3.6)**2)* param.r_spd_cv2  # 用上一步的到终点距离减去最新的到终点距离


		##############结算奖励 for 2km三路口#################
		# 求r_success
		# 计算得通过三个路口的奖励应该分别为3 20 55
		r_sucess = 0
		if Phase == 1 and self.displacement == location.endpoint and car_spd != 0 :
			r_sucess = (location.endpoint == 525)*3 + (location.endpoint == 1480)*20 + (location.endpoint == 2005)*55
			#r_sucess = 5**location.segment_id
		# if Phase == 0 and self.displacement == location.endpoint and car_spd != 0 :
		# 	r_sucess = (location.endpoint == 525)*(-3) + (location.endpoint == 1480)*(-20) + (location.endpoint == 2005)*(-55)

		# ##############结算奖励 for 师兄长距离三路口#################
		# # 求r_success
		# # 计算得通过三个路口的奖励应该分别为3 20 55
		# r_sucess = 0
		# if Phase == 1 and self.displacement == location.endpoint and car_spd != 0 :
		# 	# r_sucess = (location.endpoint == 1200)*11 + (location.endpoint == 2200)*83 + (location.endpoint == 3700)*1667 # 离散步长5米
		# 	r_sucess = (location.endpoint == 1200)*3 + (location.endpoint == 2200)*9 + (location.endpoint == 3700)*41 # 离散步长10米
		# 	# r_sucess = 5**location.segment_id
		# # if Phase == 0 and self.displacement == location.endpoint and car_spd != 0 :
		# # 	r_sucess = (location.endpoint == 525)*(-3) + (location.endpoint == 1480)*(-20) + (location.endpoint == 2005)*(-55)


		# # 求r_illegal
		# # 普锐斯模型输出I记录了模型内部的转速、加速度 越界情况；取值1 2 4 9分别对应发动机、发电机、电动机、加速度越界
		# # I为和式，如取值为3则是发动机和发电机都发生了越界
		# ci1 = param.r_ille_ci1 # 参考Automated eco-driving in urban scenarios using deep reinforcement learning 
		# ci2 = param.r_ille_ci2
		# ci3 = param.r_ille_ci3
		# r_illegal = ci1 * ((car_a > 0)*(car_a - 2.5)**2 + (car_a < 0)*(car_a + 2.5)**2) + ci2 * (I>0)*(-I) # - (car_spd == 0) * ci3
		
		self.isdone = self.isdone + r_sucess

		# 求总reward
		# r_fuel = float(r_fuel)
		# r_SOC = float(r_SOC)
		# r_spd = float(r_spd)
		# r_tra = float(r_tra)
		# r_spd = float(r_spd)
		# r_illegal = float(r_illegal)
		r_sucess = float(r_sucess)
		r = r_moving + r_sucess #+ r_illegal
		r = float(r)

		##### 存新状态 #####
		s_ = np.zeros(self.s_dim)
		s_[0] = car_spd/speedlimit
		s_[1] = abs(car_a)/2.5
		s_[2] = SOC_new
		s_[3] = self.dis2inter / (location.endpoint - location.startpoint)
		s_[4] = self.dis2termin / self.travellength
		s_[5] = Phase
		s_[6] = (Phase == 1) * remaining_time / location.GreenTiming + (Phase == 0) * remaining_time / location.RedTiming
		s_[7] = r_moving

		##### 结束标志 #####
		# done = self._get_done()
		done = False

		info = {} # 用于记录训练过程中的信息,便于观察训练状态
		info['fuel_cost'] = fuel_cost
		info['eng_spd'] = omega_list[0]
		info['eng_trq'] = T_list[0]
		info['gen_spd'] = omega_list[1]
		info['gen_trq'] = T_list[1]
		info['mot_spd'] = omega_list[2]
		info['mot_trq'] = T_list[2]
		info['eng_alp'] = alpha_list[0]
		info['gen_alp'] = alpha_list[1]
		info['mot_alp'] = alpha_list[2]
		info['car_spd'] = car_spd
		info['car_a'] = car_a
		info['SOC'] = SOC_new
		info['phase'] = Phase
		info['timing'] = remaining_time
		info['dis2inter'] = self.dis2inter
		info['displacement'] = self.displacement
		info['t'] = self.t_total
		info['dis2termin'] = self.dis2termin
		info['r'] = r
		info['speedlimit'] = speedlimit
		# info['r_fuel'] = r_fuel
		# info['r_soc'] = r_SOC
		# info['r_tra'] = r_tra
		# info['r_spd'] = r_spd
		info['r_moving'] = r_moving # 这个量写到原来r_spd_list里面
		# info['r_ill'] = r_illegal
		info['r_suc'] = r_sucess
		info['illegal'] = I

		return s_, r, done, info

	def out_info(self, info, i, action): # 将所有需要输出到csv的量append到列表并导出
		# action已经在主程序里面append了
		self.car_spd_list.append(info['car_spd'])
		self.car_a_list.append(info['car_a'])
		self.SOC_data.append(info['SOC'])
		self.Eng_spd_list.append(info['eng_spd'])
		self.Eng_trq_list.append(info['eng_trq'])
		self.Gen_spd_list.append(info['gen_spd'])
		self.Gen_trq_list.append(info['gen_trq'])
		self.Mot_spd_list.append(info['mot_spd'])
		self.Mot_trq_list.append(info['mot_trq'])
		self.Eng_alp_list.append(info['eng_alp'])
		self.Gen_alp_list.append(info['gen_alp'])
		self.Mot_alp_list.append(info['mot_alp'])

		self.Phase_list.append(info['phase'])
		self.remaining_time_list.append(info['timing'])
		self.dis2inter_list.append(info['dis2inter'])
		self.displacement_list.append(info['displacement'])
		self.t_list.append(info['t'])
		self.cost_Engine_list.append(info['fuel_cost'])
		self.Reward_list_all.append(info['r'])
		self.I_list.append(info['illegal'])

		# self.r_fuel_list.append(info['r_fuel'])
		# self.r_soc_list.append(info['r_soc'])
		# self.r_tra_list.append(info['r_tra'])
		# self.r_spd_list.append(info['r_spd'])
		self.r_spd_list.append(info['r_moving'])
		# self.r_illegal_list.append(info['r_ill'])
		self.r_success_list.append(info['r_suc'])
	
	def write_info(self, i):
		filename = "/eposide " + str(i) + ".csv"
		f = open(data_dir + filename ,'w')
		csv_write = csv.writer(f)
		csv_write.writerow(self.action_list)
		csv_write.writerow(self.Eng_trq_list)
		csv_write.writerow(self.Eng_alp_list)
		csv_write.writerow(self.Eng_spd_list)
		csv_write.writerow(self.Gen_trq_list)
		csv_write.writerow(self.Gen_alp_list)
		csv_write.writerow(self.Gen_spd_list)
		csv_write.writerow(self.Mot_trq_list)
		csv_write.writerow(self.Mot_alp_list)
		csv_write.writerow(self.Mot_spd_list)
		csv_write.writerow(self.car_spd_list)
		csv_write.writerow(self.car_a_list)
		csv_write.writerow(self.SOC_data)
		csv_write.writerow(self.I_list)
		csv_write.writerow(self.displacement_list)
		# csv_write.writerow(signal_flag_list)
		csv_write.writerow(self.t_list)
		csv_write.writerow(self.Phase_list)
		csv_write.writerow(self.remaining_time_list)
		csv_write.writerow(self.Reward_list_all)
		csv_write.writerow(self.cost_Engine_list)

		# csv_write.writerow(self.r_fuel_list)
		# csv_write.writerow(self.r_soc_list)
		# csv_write.writerow(self.r_tra_list)
		csv_write.writerow(self.r_spd_list)
		# csv_write.writerow(self.r_illegal_list)
		csv_write.writerow(self.r_success_list)
		f.close()




		



