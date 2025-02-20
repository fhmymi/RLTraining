import numpy as np
import gymnasium as gym

NUM_DIGITIZED = 6


class QTableAgent:
    def __init__(self, action_space, n_states, eta=0.5, gamma=0.8, NUM_DIGITIZED=6):
        self.action_space = action_space
        self.n_states = n_states
        self.eta = eta
        self.gamma = gamma
        self.NUM_DIGITIZED = NUM_DIGITIZED
        self.q_table = np.random.uniform(low=-1, high=1, size=(NUM_DIGITIZED ** self.n_states, self.action_space))

    def q_learning(self, state, action, reward, next_state):
        obs_ind = QTableAgent.digitize_state(state, self.NUM_DIGITIZED)
        obs_next_ind = QTableAgent.digitize_state(next_state, self.NUM_DIGITIZED)
        self.q_table[obs_ind, action] += self.eta * (reward + self.gamma * np.max(self.q_table[obs_next_ind, :]) - self.q_table[obs_ind, action])
    def choose_action(self, state, episode):
        eps = 0.5 / (episode + 1)
        obs_ind = QTableAgent.digitize_state(state, self.NUM_DIGITIZED)
        if np.random.rand() <= eps:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[obs_ind, :])

    # 分桶， 5个值，对应 6 个分段，即 6 个桶 (0, 1, 2, 3, 4, 5)
    @staticmethod
    def bins(clip_min, clip_max, num_bins=6):
        return np.linspace(clip_min, clip_max, num_bins + 1)[1:-1]

    # 按 6 进制映射将 4位 6 进制数映射为 id，
    @staticmethod
    def digitize_state(observation, NUM_DIGITIZED):
        pos, cart_v, angle, pole_v = observation
        digitized = [np.digitize(pos, bins=QTableAgent.bins(-2.4, 2.4, NUM_DIGITIZED)),
                     np.digitize(cart_v, bins=QTableAgent.bins(-3., 3, NUM_DIGITIZED)),
                     np.digitize(angle, bins=QTableAgent.bins(-0.418, 0.418, NUM_DIGITIZED)),
                     np.digitize(pole_v, bins=QTableAgent.bins(-2, 2, NUM_DIGITIZED))]
        # 3,1,2,4 (4位10进制数) = 4*10^0 + 2*10^1 + 1*10^2 + 3*10^3，最终的取值范围是 0-9999，总计 10^4 == 10000
        # a,b,c,d (4位6进制数) = d*6^0 + c*6^1 + b*6^2 + a*6^3，最终的取值范围是 0-`5555`(1295)，总计 6^4 == 1296
        ind = sum([d * (NUM_DIGITIZED ** i) for i, d in enumerate(digitized)])
        return ind


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
        last_step = 0
        step = 0
        done = False
        while not done:
            env.render()
            action = agent.choose_action(state, episode)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode: {episode} | Step: {step} | Continue Success Episodes: {continue_success_episodes}")
                # if step < last_step:
                #     if step < 50:
                #         reward = -1
                #     else:
                #         reward = 0.05
                # else:
                #     reward = 1
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
