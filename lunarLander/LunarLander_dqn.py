import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, input_size=8, output_size=4):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return F.softmax(self.fc3(x), dim=1)


def train(env, policy, optimizer, gamma=0.99):
    # 数据收集
    states = []
    actions = []
    rewards = []
    log_probs = []

    state, _ = env.reset()
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        # 存储数据
        states.append(state)
        actions.append(action)
        log_probs.append(m.log_prob(action))
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            # 计算折扣回报
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

            # 计算损失
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)

            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

            break


if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)

    for episode in range(1000):
        train(env, policy, optimizer)
        if episode % 50 == 0:
            print(f"Episode {episode} completed")
    torch.save(policy.state_dict(), './LunarLander_policy.pth')
    env.close()
