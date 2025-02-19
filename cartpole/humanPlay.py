import time
import gym
from pynput import keyboard

# 初始化CartPole环境
env = gym.make('CartPole-v1', render_mode='human',new_step_api=True)
env.reset()

action = 0 # 默认动作，0表示向左，1表示向右

def on_press(key):
    global action
    try:
        if key.char == 'a': # 按下'a'键向左移动
            action = 0
            print('向左移动')
        elif key.char == 'd': # 按下'd'键向右移动
            action = 1
            print('向右移动')
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc: # 按下'Esc'键退出游戏
        return False

# 开始监听键盘事件
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    env.render() # 渲染画面
    observation, reward, done, truncated, info = env.step(action) # 执行动作
    time.sleep(0.1)
    if done:
        print("Game Over!")
        break

listener.stop()
env.close()