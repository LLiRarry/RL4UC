# This is a sample Python script.
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, DDPG, TD3
from env import *


def prepare_env():
    mode = 'train'
    get_info = json.load(open('../Data/data.json'))
    env_uc = UC_Env(gen_info=get_info, mode=mode)
    env = DummyVecEnv([lambda: env_uc])
    return env


def train_model(env, RL_model='PPO'):
    if RL_model == 'A2C':
        model = A2C("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        return model
    elif RL_model == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        return model
    elif RL_model == 'DDPG':
        model = DDPG("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        return model
    elif RL_model == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50000)
        return model

def plot_rewards(rewards, model_names):
    plt.figure(figsize=(10, 6))

    for i, reward in enumerate(rewards):
        plt.plot(np.arange(len(reward)), reward, label=model_names[i])

    plt.title('Reward Curves for Reinforcement Learning Models')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()


model_names = ['A2C', 'PPO', 'DDPG', 'TD3']
env = prepare_env()
rewards = []

for model_name in model_names:
    model = train_model(env, RL_model=model_name)
    episode_rewards = []
    obs = env.reset()

    for _ in range(48):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        if done:
            obs = env.reset()

    rewards.append(episode_rewards)

plot_rewards(rewards, model_names)

