# -*- coding: utf-8 -*-
"""
DDPG_Prius
"""
#import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from ddpg import *

import math
import os
#import tensorflow.compat.v1 as tf
import tensorflow as tf
#from tensorflow_core.contrib.distribution.python.ops.bijectors import batch_normalization as batch_norm
import numpy as np 
import gym
import scipy.io as scio
import matplotlib.pyplot as plt
from Priority_Replay import Memory

from Parameter import Param
param = Param()
from environment import Env
from Prius_model_new import Prius_model
from Fuel_Consumption import get_fuel
from State_Machine import reward_traffic
import PlotEnv as ple
from calculateBestOmega import calculate_speed
from Road import Road
import csv

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

#####################  hyper parameters  ####################
exploration_decay_start_step = param.exploration_decay_start_step
is_training = True
MAX_EPISODES = param.MAX_EPISODE

#####################  Training Process  ####################
##################### 导入环境，给定参数 ########################
env = Env()
a_bound = 1
DDPG = DDPG(env.s_dim, env.a_dim)
past_action = np.array([0., 0., 0.])
# control exploration
var = param.var

if is_training:
    print("Under Training")

    # 开始训练大循环
    for i in range(MAX_EPISODES):
        # print("调试：开始大循环")
        total_step = 0
        step_episode = 0
        mean_reward_all = 0
        cost_engine = 0
        ep_reward_all = 0
        sum_r_fuel = 0
        sum_r_soc = 0
        sum_r_tra = 0
        sum_r_spd = 0
        sum_r_ill = 0
        sum_r_suc = 0
        state, omega_list = env.reset() # 获得初始状态

        # 开始小循环，逐个离散步长更新
        for j in range(int(env.travellength/env.STEP_SIZE)):
            # print("调试：进入小循环",str(j))
            T_list = []
            action = DDPG.action(state)
            env.action_list.append(action)
            # a = np.clip(np.random.laplace(action, var), 0, 1)                         # 动作归一化0到1 变成了一个概率
            # Eng_pwr_opt = (a[0]) * 56000                                              # optimal curve约束在最大pwr是56000 -> 到你这里要注意约束电机和发动机转矩，让动作 = 概率 * 转矩上限
            a1 = np.clip(np.random.laplace(action[0], var), 0., 1.)
            Eng_trq = (a1) * 120
            T_list.append(Eng_trq)

            a2 = np.clip(np.random.laplace(action[1], var), -1., 1.)
            Gen_trq = (a2) * 75
            T_list.append(Gen_trq)

            a3 = np.clip(np.random.laplace(action[2], var), -1., 1.)
            Mot_trq = (a3) * 400
            T_list.append(Mot_trq)

            # 更新步长末状态，reward
            s_new, r, done, info = env.step(T_list, state, omega_list)

            # 步长内油耗提取
            fuel_cost = info['fuel_cost']

            # 更新经验缓存区
            # print("===============")
            # print("s_new = ",s_new)
            # print("state = ",state)
            # print("action = ",action)
            # print("r = ",r)
            # print("is done? :",done)
            time_step = DDPG.perceive(state, action, r, s_new, done)

            if time_step % param.var_update_step == 0 and time_step > exploration_decay_start_step:
                var *= param.var_multi

            # 更新末状态为下一个步长的初状态
            state = s_new

            # 输出过程数据到列表
            env.out_info(info, i, action)

            # Episode内步长累计
            cost_engine += fuel_cost
            ep_reward_all += r
            # sum_r_fuel += info['r_fuel']
            # sum_r_soc += info['r_soc']
            # sum_r_tra += info['r_tra']
            sum_r_spd += info['r_moving']
            # sum_r_ill += info['r_ill']
            sum_r_suc += info['r_suc']
            total_step += 1

            # 终端输出episode信息
            if j == int(env.travellength/env.STEP_SIZE)-1:
                # print("调试：进来了吧？")
                # 输出SOC
                SOC_final = state[2]
                # 计算油耗 单位L
                cost_engine = cost_engine/0.72/1000
                # 计算总里程
                total_mileage = env.displacement/1000 # 单位Km
                # 计算百公里油耗
                cost_engine_100km = 100 * cost_engine/total_mileage

                # 计算平均reward
                mean_reward = ep_reward_all / (env.travellength/env.STEP_SIZE)
                env.mean_reward_list.append(mean_reward)
                # mean_r_fuel = sum_r_fuel / (env.travellength/env.STEP_SIZE)
                # env.mean_fuel_list.append(mean_r_fuel)
                # mean_r_soc = sum_r_soc / (env.travellength/env.STEP_SIZE)
                # env.mean_soc_list.append(mean_r_soc)
                # mean_r_tra = sum_r_tra / (env.travellength/env.STEP_SIZE)
                # env.mean_tra_list.append(mean_r_tra)
                mean_r_spd = sum_r_spd / (env.travellength/env.STEP_SIZE)
                env.mean_spd_list.append(mean_r_spd)
                # mean_r_ill = sum_r_ill / (env.travellength/env.STEP_SIZE)
                # env.mean_ill_list.append(mean_r_ill)
                mean_r_suc = sum_r_suc / (env.travellength/env.STEP_SIZE)
                env.mean_suc_list.append(mean_r_suc)

                print('Episode:', i, ' cost_Engine: %.3f' % cost_engine, ' Fuel_100Km: %.3f' % cost_engine_100km, ' SOC-final: %.3f' % SOC_final, ' Explore: %.4f' % var)
        
        # 列表写入文件
        if (i % param.write == 0) and (i > 0):
            env.write_info(i)

        # 每n个episode 画第n个episode里车辆的位移时间分布图 和 SOC变化
        if (i % param.pltenv == 0) & (i > 0) :
            ple.initEnv(env.t_list, env.displacement_list, i)
            print("画图函数被成功调用，并且画完了！保存起来了！")
            temp_soc_data = env.SOC_data
            del temp_soc_data[0] # 把SOC列表的第一位“SOC”字符串删去（按位索引删除）
            x = np.arange(0, len(temp_soc_data), 1)
            y = temp_soc_data
            plt.plot(x, y)
            plt.xlabel('distance')
            plt.ylabel('SOC')
            picname = SOC_dir + "/episode" + str(i) + "_SOC.png"
            plt.savefig(picname, dpi = 500, bbox_inches='tight')
            # 自动调整坐标轴
            plt.tight_layout()
            plt.close()     
        # DRL-EMS源码未直接使用list_even和list_odd 
        # # mean_reward记录了一个eposide里的情况，mean_reward_all在小循坏之外初始化的，对多个episode的平均回报累加
        # mean_reward_all += mean_reward   
        # if (step_episode % 10) == 0 and step_episode >= 10:
        #     if (step_episode / 10) % 2 == 0:
        #         env.list_even.append(mean_reward_all)
        #     else:
        #         env.list_odd.append(mean_reward_all)
        #     mean_reward_all = 0 

        # 每100个episode，画这100次Episode的平均reward变化
        if (i % param.pltreward == 0) & (i > 0):
            x = np.arange(0, len(env.mean_reward_list), 1)
            y = env.mean_reward_list
            # print(y)
            plt.plot(x, y)
            plt.xlabel('Epsoide')
            plt.ylabel('Mean_Reward')
            plt.savefig(reward_dir + '/buffersize-100W_Mean_reward' + str(i) + '.png', dpi = 500, bbox_inches='tight')
            plt.close()
        if (i % param.pltreward == 0) & (i > 0):
            x = np.arange(0, len(env.mean_reward_list), 1)
            # y_fuel = env.mean_fuel_list
            # y_soc = env.mean_soc_list
            # y_tra = env.mean_tra_list
            y_spd = env.mean_spd_list
            # y_ill = env.mean_ill_list
            y_suc = env.mean_suc_list
            # line_fuel, = plt.plot(x, y_fuel, linewidth = '1', linestyle = '-', label ='r_fuel', c = 'orange')
            # line_soc, = plt.plot(x, y_soc, linewidth = '1', linestyle = '-', label ='r_soc', c = 'blue')
            # line_tra, = plt.plot(x, y_tra, linewidth = '1', linestyle = '-', label ='r_tra', c = 'y')
            line_spd, = plt.plot(x, y_spd, linewidth = '1', linestyle = '-', label ='r_spd', c = 'black')
            # line_ill, = plt.plot(x, y_ill, linewidth = '1', linestyle = '-', label ='r_ill', c = 'red')
            line_suc, = plt.plot(x, y_suc, linewidth = '1', linestyle = '-', label ='r_suc', c = 'green')
            plt.xlabel('Epsoide')
            plt.ylabel('Mean_Reward')
            # plt.legend([line_fuel, line_soc, line_tra, line_spd, line_ill, line_suc], ['r_fuel', 'r_soc', 'r_tra', 'r_spd', 'r_ill', 'r_suc'], loc = 'lower right')
            # plt.legend([line_spd, line_ill, line_suc], ['r_spd', 'r_ill', 'r_suc'], loc = 'lower right')
            plt.legend([line_spd, line_suc], ['r_spd', 'r_suc'], loc = 'lower right')
            plt.savefig(reward_dir + '/各项reward平均值变化' + str(i) + '.png', dpi = 500, bbox_inches='tight')
            plt.close()

        # # 当探索系数var足够小，学习效果不再发生较大变化时，退出
        # if var < param.var_threshold :
        #     print("探索值小于0.009，可以停止训练")
        #     break

        # # 有成功绿波，就存一次模型
        if env.isdone == 78 and average_spd <= 20 :
            print("第",i,"episode, ""绿波通过成功")
            DDPG.actor_network.save_network(DDPG.time_step)
            DDPG.critic_network.save_network(DDPG.time_step)

        # # 当成功绿波通过所有路口，训练退出
        del env.car_spd_list[0]
        average_spd = np.mean(env.car_spd_list)
        if env.isdone == 78 and average_spd <= 20 and var < param.var_threshold:
            print("智能体绿波通过了所有路口，训练推出")
            DDPG.actor_network.save_network(DDPG.time_step)
            DDPG.critic_network.save_network(DDPG.time_step)
            break

    # 到这里，所有的训练结束了，训练了全部的eposide

    # 跳出训练后画运动轨迹
    ple.initEnv(env.t_list, env.displacement_list, i)
    print("画图函数被成功调用，并且画完了！保存起来了！")
    
    # 画最后一次的全程SOC变化曲线
    # mean_discrepancy_list = list(map(lambda x, y: y - x, list_even, list_odd))  
    del env.SOC_data[0] # 把SOC列表的第一位“SOC”字符串删去（按位索引删除）
    x = np.arange(0, len(env.SOC_data), 1)
    y = env.SOC_data
    plt.plot(x, y)
    plt.xlabel('distance')
    plt.ylabel('SOC')
    # plt.show()
    plt.savefig('SOC STATE.png', dpi = 500, bbox_inches='tight')
    # 自动调整坐标轴
    plt.tight_layout()
    plt.close()

    # 训练reward变化
    x = np.arange(0, len(env.mean_reward_list), 1)
    y = env.mean_reward_list
    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Mean_Reward')
    plt.savefig('Mean_reward.png', dpi = 500, bbox_inches='tight')
    plt.close()

else:
    print("Testing Mode")
    test_time = 64
    for j in range(test_time):
        total_step = 0
        step_episode = 0
        mean_reward_all = 0
        cost_engine = 0
        state, omega_list = env.reset() # 获得初始状态
        for i in range(int(env.travellength/env.STEP_SIZE)):
            T_list = []
            action = DDPG.action(state)
            env.action_list.append(action)
            var = 0.1691
            # a = np.clip(np.random.laplace(action, var), 0, 1)                         # 动作归一化0到1 变成了一个概率
            # Eng_pwr_opt = (a[0]) * 56000                                              # optimal curve约束在最大pwr是56000 -> 到你这里要注意约束电机和发动机转矩，让动作 = 概率 * 转矩上限
            a1 = np.clip(np.random.laplace(action[0], var), 0., 1.)
            Eng_trq = (a1) * 120
            T_list.append(Eng_trq)

            a2 = np.clip(np.random.laplace(action[1], var), -1., 1.)
            Gen_trq = (a2) * 75
            T_list.append(Gen_trq)

            a3 = np.clip(np.random.laplace(action[2], var), -1., 1.)
            Mot_trq = (a3) * 400
            T_list.append(Mot_trq)

            # 环境交互
            s_new, r, done, info = env.step(T_list, state, omega_list)

            # 步长内油耗提取
            fuel_cost = info['fuel_cost']

            # 更新末状态为下一个步长的初状态
            state = s_new

            # 输出过程数据到列表
            env.out_info(info, i, action)
    
        # 跑完全程，把数据写入文档
        print("测试，第",j,"次")
        env.write_info(j)

        # 画轨迹
        ple.initEnv(env.t_list, env.displacement_list, j)
        print("画图函数被成功调用，并且画完了！保存起来了！")
    







            




