import BrainDQN_Nature as bn
import gym
import tensorflow as tf

def main():
    config = bn.Config()

    env = gym.make('Breakout-v0')
    print(env.action_space) # 查看这个环境中可用的 action 有多少个
    print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
    config.actions = env.action_space.n
    brain = bn.BrainDQN(config)

    for i_episode in range(200000):

        # 获取回合 i_episode 第一个 observation
        observation = env.reset()
        brain.setInitState(observation)  

        while True:
            # time.sleep(0.1)
            env.render()    # 刷新环境

            action = brain.getAction()# 选行为

            observation_, reward, done, _ = env.step(action) # 获取下一个 state

            brain.setPerception(observation_,action,reward,done)
            

            if done:
                break




if __name__ == '__main__':
    main()