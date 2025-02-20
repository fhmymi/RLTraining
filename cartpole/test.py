import gymnasium as gym
import random
import numpy as np

Num = 10
rate = 0.5
factor = 0.9

p_bound = np.linspace(-2.4, 2.4, Num - 1)
v_bound = np.linspace(-3, 3, Num - 1)
ang_bound = np.linspace(-0.5, 0.5, Num - 1)
angv_bound = np.linspace(-2.0, 2.0, Num - 1)


def state_dig(state):  # 离散化
    p, v, ang, angv = state
    digital_state = (np.digitize(p, p_bound),
                     np.digitize(v, v_bound),
                     np.digitize(ang, ang_bound),
                     np.digitize(angv, angv_bound))
    return digital_state


if __name__ == '__main__':

    env = gym.make('CartPole-v1', render_mode="rgb_array")

    action_space_dim = env.action_space.n
    q_table = np.zeros((Num, Num, Num, Num, action_space_dim))

    for i in range(3000):
        state, _ = env.reset()
        digital_state = state_dig(state)

        step = 0
        gameOver = False
        while not gameOver:
            if i % 10 == 0:
                env.render()

            step += 1
            epsi = 1.0 / (i + 1)
            if random.random() < epsi:
                action = random.randrange(action_space_dim)
            else:
                action = np.argmax(q_table[digital_state])

            next_state, reward, done, truncated, _ = env.step(action)
            next_digital_state = state_dig(next_state)

            if done:
                if step < 200:
                    reward = -1
                else:
                    reward = 1
            else:
                reward = 0

            current_q = q_table[digital_state][action]  # 根据公式更新qtable
            q_table[digital_state][action] += rate * (reward + factor * max(q_table[next_digital_state]) - current_q)

            digital_state = next_digital_state

            if done:
                print(step)
                break
