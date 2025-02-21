import time
import gymnasium as gym
from pynput import keyboard

# 初始化CartPole环境
env = gym.make("LunarLander-v3", render_mode="human")
env.reset()

action = 0  # 默认动作，0表示向左，1表示向右


def on_press(key):
    global action
    try:
        if key == keyboard.KeyCode.from_char('w') or key == keyboard.Key.up:
            action = 0
        elif key == keyboard.KeyCode.from_char('a') or key == keyboard.Key.left:
            action = 3
        elif key == keyboard.KeyCode.from_char('s') or key == keyboard.Key.down:
            action = 2
        elif key == keyboard.KeyCode.from_char('d') or key == keyboard.Key.right:
            action = 1

    except AttributeError:
        pass


def on_release(key):
    if key == keyboard.Key.esc:  # 按下'Esc'键退出游戏
        return False


# 开始监听键盘事件
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    env.render()  # 渲染画面
    observation, reward, done, truncated, info = env.step(action)  # 执行动作
    time.sleep(0.1)
    if done:
        print("Game Over!")
        break

listener.stop()
env.close()
