import gym # 导入 Gym 的 Python 接口环境包
env = gym.make('CartPole-v1', render_mode='human',new_step_api=True) # 构建实验环境
env.reset() # 重置一个回合
for _ in range(10):
    env.reset()
    env.render()
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(observation, reward, done, info)
env.close()