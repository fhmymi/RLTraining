import time

import torch
from torch.distributions import Categorical
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
class CartPolePolicy(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.drop = nn.Dropout(p=0.27)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def compute_loss(n, log_p):
    r = list()
    for i in range(n, 0, -1):
        r.append(i * 1.0)
    r = torch.tensor(r)
    r = (r - r.mean()) / (r.std() + 1e-5)
    loss = 0
    for pi, ri in zip(log_p, r):
        loss += -pi * ri
    return loss


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state1, _ = env.reset()

    policy = CartPolePolicy()
    #如果存在模型，则加载模型
    if torch.cuda.is_available():
        policy.load_state_dict(torch.load('./cartpole_policy.pth'))
        print('加载模型成功', "use_cuda")
    else:
        policy.load_state_dict(torch.load('./cartpole_policy.pth', map_location=torch.device('cpu')))
        print('加载模型成功', "use_cpu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0027)
    log_chart = []
    max_episodes = 500
    max_step = 0
    epi_tqdm = tqdm(range(1, max_episodes + 1))
    continue_success_episodes = 0
    for episode in epi_tqdm:
        state, _ = env.reset()
        step = 0
        log_p = list()
        done = False
        while not done:
            step += 1
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy(state)
            m = Categorical(action_probs)
            action = m.sample()
            state, reward, terminated, truncated, info = env.step(action.item())
            if terminated or truncated:
                done = True
                if step > max_step:
                    max_step = step
                if step >= 500:
                    continue_success_episodes += 1
                else:
                    continue_success_episodes = 0
                log_chart.append((episode, step))
            log_p.append(m.log_prob(action))

        optimizer.zero_grad()
        loss = compute_loss(step, log_p)
        loss.backward()
        optimizer.step()

        epi_tqdm.set_description(f"Episode {episode} Run steps: {step}| Max steps:{max_step}| CSE: {continue_success_episodes}")
    torch.save(policy.state_dict(), './cartpole_policy.pth')
    plt.plot([x[0] for x in log_chart], [x[1] for x in log_chart])
    # plt.yscale('log')  # 设置y轴为对数刻度
    plt.show()
