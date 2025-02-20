import random

import numpy as np
import gymnasium as gym

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

class QTableAgent:
    def __init__(self, action_space, n_states, NUM_DIGITIZED=Num):
        self.action_space = action_space
        self.n_states = n_states
        self.NUM_DIGITIZED = NUM_DIGITIZED
        self.q_table = np.zeros((NUM_DIGITIZED, NUM_DIGITIZED, NUM_DIGITIZED, NUM_DIGITIZED, action_space))

    def q_learning(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        self.q_table[state][action] += rate * (reward + factor * max(self.q_table[next_state]) - current_q)

    def choose_action(self, state, episode):
        eps = 1 / (episode + 1)
        delta = random.random()
        # print(f"本次生成的delta：{delta}")
        if delta <= eps:
            action = random.randrange(self.action_space)
        else:
            action = np.argmax(self.q_table[state])
        return action

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.reset()
    action_space = env.action_space.n
    n_states = env.observation_space.shape[0]
    agent = QTableAgent(action_space, n_states)

    max_episodes = 1000
    max_step = 1000

    continue_success_episodes = 0

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state_dig(state)
        last_step = 0
        step = 0
        done = False
        while not done:
            env.render()
            action = agent.choose_action(state, episode)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = state_dig(next_state)
            if terminated or truncated:
                print(f"Episode: {episode} | Step: {step} | Continue Success Episodes: {continue_success_episodes}")
                if step < 200:
                    reward = -1
                else:
                    reward = 1

                if step < 200:
                    continue_success_episodes = 0
                else:
                    continue_success_episodes += 1

                last_step = step
                break
            else:
                reward = 0
            agent.q_learning(state, action, reward, next_state)

            state = next_state

            step += 1
            done = terminated or truncated
