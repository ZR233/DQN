import gym
import DeepQNetwork as dq
import time

env = gym.make('Breakout-v0')
print(env.action_space) # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个

config = dq.Config()
# config.n_x = env.observation_space.shape[0]
# config.n_y = env.observation_space.shape[1]
# config.n_z = env.observation_space.shape[2]
config.n_actions=env.action_space.n

dq.train(config, env)