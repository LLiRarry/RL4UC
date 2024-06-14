import numpy as np
import pandas as pd
import os
import json
import torch
# sigmod() 输出范围0-1，反归一化到功率区间
def inverse_min_max_scaling_single(scaled_value, original_min, original_max):
    original_value = (scaled_value - 0) / (1 - 0) * (original_max - original_min) + original_min
    return original_value
class UC_Env(object):
    def __init__(self, gen_info, mode='train'):
        self.mode = mode
        self.gen_info = gen_info  # the basic information
        self.num_gen = 8
        self.episode_length = 24
        self.max_output = np.array(self.gen_info['power_h'])
        self.min_output = np.array(self.gen_info['power_l'])
        self.t_min_down = np.array(self.gen_info['min_down'])
        self.t_min_up = np.array(self.gen_info['min_up'])
        self.product_cost = np.array(self.gen_info['product_cost'])
        self.start_up_cost = np.array(self.gen_info['start_up_cost'])
        self.shutdown_cost = np.array(self.gen_info['shutdown_cost'])
        self.load_demand = np.array(self.gen_info['load_demand'])
        self.status = np.array([0 for _ in range(self.num_gen)])
        self.infeasible = False
        self.action_size = self.num_gen
        self.time_step=0

    def _update_gen_status(self, action):  # action_o
        def single_update(status, action):
            # 如果已经开启了一段时间
            if status > 0:
                if action == 1:
                    return (status + 1)
                else:
                    return -1
            # 如果已经关闭了一段时间
            else:
                if action == 1:
                    return 1
                else:
                    return (status - 1)

        self.status = np.array([single_update(self.status[i], action[i]) for i in range(len(self.status))])

    def _legalise_action(self, action):  # 最短启停时间约束，这里采用一个mask机制
        # 这是还没有更新的status
        self.must_on = np.array([True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array(
            [True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])
        # 异或逻辑mask掉不合法的开停动作
        x = np.logical_or(action, self.must_on)
        x = x * np.logical_not(self.must_off)
        # 计算开停的惩罚项
        punishment = 0
        for i in range(self.num_gen):
            if action[i] == 1 and self.must_off[i]:
                punishment += 1
            if action[i] == 0 and self.must_on[i]:
                punishment += 1
        return np.array(x, dtype=int), punishment

    def _is_legal_on_time_constrain(self, action):
        action = np.array(action)
        illegal_on = np.any(action[self.must_on] == 0)
        illegal_off = np.any(action[self.must_off] == 1)
        if any([illegal_on, illegal_off]):
            return False
        else:
            return True

    def _is_satisfiy_load(self, action, power_output):  # damand 约束
        delta = 0
        if np.sum(action* power_output) > self.load_demand[self.time_step]:
            is_s = True
        else:
            # 缺少的能量作为惩罚项
            delta = self.load_demand[self.time_step] - np.sum((action * power_output))
            is_s = False
        return is_s, delta

    def _get_reward(self, action,power_output):
        genaration_cost = np.sum(action*power_output* self.product_cost)
        start_cost = 0
        shutdown_cost = 0
        for i in range(self.num_gen):
            if self.status[i] > 0 and action[i] == 0:
                shutdown_cost += self.shutdown_cost[i]
            elif self.status[i] < 0 and action[i] == 1:
                start_cost += self.start_up_cost[i]
        sum_cost = genaration_cost + start_cost + shutdown_cost
        return (-1) * sum_cost

    def step(self, action):
        (actions_discrete, actions_continuous)=action
        action_on_off = np.array(actions_discrete)
        power_output=np.clip(actions_continuous, 0, 1)
        power_output=self.min_output+(self.max_output-self.min_output)*power_output
        legal_action, punish_onoff = self._legalise_action(action_on_off)
        is_legal, delta = self._is_satisfiy_load(legal_action, self.time_step)
        if is_legal:
            self.infeasible = False
            reward = self._get_reward(action_on_off,power_output) - punish_onoff
        else:
            self.infeasible = True
            reward = self._get_reward(action_on_off,power_output) - punish_onoff*1000 -delta*100
        self._update_gen_status(action_on_off)  # 这里再更新机组状态
        next_states = np.concatenate((self.status.flatten(), np.array([self.load_demand[self.time_step]])), axis=0)
        if self.time_step == self.episode_length - 1:
            done = True
            self.status = self.t_min_up  # or （-1）* self.t_min_down 都可以，表示初始状态不受前面开停状态约束
            self.infeasible = False
            self.time_step = 0
            status = np.concatenate((self.status, np.array([self.load_demand[0]])), axis=0)
            status_time = {'status': status, 'timestep': self.time_step}
            obs_new = np.concatenate((status_time['status'], [status_time['timestep']]))
            return obs_new, reward, done
        else:
            done = False
            status_time = {'status': next_states, 'timestep': self.time_step}
            obs_new = np.concatenate((status_time['status'], [status_time['timestep']]))
            self.time_step += 1
            return obs_new, reward, done

    def reset(self):
        # 这里只选用前一时刻状态、下一时刻需求以及时间周期作为状态输入
        self.status = self.t_min_up  # or （-1）* self.t_min_down 都可以，表示初始状态不受前面开停状态约束
        self.infeasible = False
        self.time_step=0
        status=np.concatenate((self.status, np.array([self.load_demand[0]])), axis=0)
        status_time = {'status': status, 'timestep': self.time_step}
        obs_new = np.concatenate((status_time['status'], [status_time['timestep']]))
        return obs_new
def Make_UC_env(mode='train'):
    get_info = json.load(open('../Data/data.json'))
    env = UC_Env(gen_info=get_info, mode=mode)
    env.reset()
    return env
