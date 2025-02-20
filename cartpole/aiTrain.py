import time

import torch
from torch.distributions import Categorical
from torch import nn
import torch.nn.functional as F
import gymnasium as gym


class CartPolePolicy(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.drop = nn.Dropout(p=0.4)

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
    env = gym.make('CartPole-v1', render_mode='human')
    state1, _ = env.reset(seed=543)
    torch.manual_seed(543)

    policy = CartPolePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    max_episodes = 100
    max_action = 500
    max_step = 5000

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        step = 0
        log_p = list()
        for step in range(1, max_step + 1):
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy(state)
            m = Categorical(action_probs)
            action = m.sample()
            state, reward, terminated, truncated, info = env.step(action.item())
            if terminated:
                print(f"Episode {episode} finished after {step} timesteps")
                break
            log_p.append(m.log_prob(action))
        if step > max_step:
            print(f"Episode {episode} clear the stage")
            break

        optimizer.zero_grad()
        loss = compute_loss(step, log_p)
        loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print(f"Episode {episode} Run steps: {step}")
    torch.save(policy.state_dict(), 'cartpole_policy.pth')
