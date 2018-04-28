import gym
from DeepQNetwork import DeepQNetwork
import time

env = gym.make('Breakout-v0')   # 定义使用 gym 库中的那一个环境
# env = env.unwrapped # 不做这个会有很多限制

print(env.action_space) # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个

# 定义使用 DQN 的算法
RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_x=env.observation_space.shape[0],
                  n_y=env.observation_space.shape[1],
                  n_z=env.observation_space.shape[2],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=50,
                  e_greedy_increment=0.0008,)

total_steps = 0 # 记录步数

for i_episode in range(1000):

    # 获取回合 i_episode 第一个 observation
    observation = env.reset()
    ep_r = 0
    while True:
        # time.sleep(0.1)
        env.render()    # 刷新环境

        action = RL.choose_action(observation)  # 选行为

        observation_, reward, done, info = env.step(action) # 获取下一个 state
        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 100:
            RL.learn()  # 学习

        ep_r += reward
        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
        # 最后输出 cost 曲线
RL.plot_cost()
