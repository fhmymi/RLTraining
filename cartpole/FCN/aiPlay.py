import time

import torch
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
    for i in range(n,0,-1):
        r.append(i * 1.0)
    r = torch.tensor(r)
    r = (r-r.mean()) / (r.std()+1e-5)
    loss = 0
    for pi ,ri in zip(log_p,r):
        loss += -pi * ri
    return loss

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    policy = CartPolePolicy()
    policy.load_state_dict(torch.load('./cartpole_policy.pth'))  # 注意文件路径
    policy.eval()

    start_time = time.time()
    max_action = 1000
    step = 0
    done = False
    while not done:
        env.render()
        step += 1
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = policy(state)
        action = torch.argmax(action_probs, dim=1).item()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            print(f"Game Over! played time {time.time() - start_time}", f'step:{step}, terminated:{terminated}, truncated:{truncated}, info:{info},state:{state}')