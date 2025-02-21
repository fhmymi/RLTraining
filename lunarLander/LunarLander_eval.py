import gymnasium as gym
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self, input_size=8, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        return torch.softmax(self.fc3(x), dim=1)


def evaluate_model(model_path, num_episodes=100, render_sample=False):
    # 环境初始化
    env = gym.make("LunarLander-v3", render_mode="human")
    policy = Policy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    # 评估指标
    total_rewards = []
    success_count = 0
    fuel_consumptions = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        terminated = False
        truncated = False

        # 样本演示渲染（每10次展示1次）
        if render_sample and ep % 10 == 0:
            demo_env = gym.make("LunarLander-v3", render_mode="human")
            demo_env.reset()
            frames = []

        while not (terminated or truncated):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()

            next_state, reward, terminated, truncated, info = env.step(action)

            # 记录关键指标
            episode_reward += reward
            step_count += 1
            if 'fuel' in info:  # 如果环境提供燃料消耗数据
                fuel_consumptions.append(info['fuel'])

            # 采集演示帧
            if render_sample and ep % 10 == 0:
                frames.append(demo_env.render())

            state = next_state

        # 记录成功着陆
        if terminated and abs(state) < 0.5:  # 根据实际着陆条件调整
            success_count += 1

        total_rewards.append(episode_reward)

        # 生成样本演示视频
        if render_sample and ep % 10 == 0:
            demo_env.close()
            save_demo(frames, f"demo_episode_{ep}.gif")

    # 性能分析
    metrics = {
        "average_reward": np.mean(total_rewards),
        "success_rate": success_count / num_episodes,
        "reward_std": np.std(total_rewards),
        "fuel_per_step": np.mean(fuel_consumptions) if fuel_consumptions else 0,
        "reward_trend": total_rewards  # 用于绘制学习曲线
    }

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["reward_trend"], alpha=0.6)
    plt.title("Reward Distribution Across Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("evaluation_plot.png")

    env.close()
    return metrics


def save_demo(frames, filename):
    import imageio
    # 优化GIF生成参数
    imageio.mimsave(filename, frames, fps=30,
                    optimize=True,
                    quantizer='wu',
                    duration=50)


if __name__ == "__main__":
    metrics = evaluate_model(
        model_path="./LunarLander_policy.pth",
        num_episodes=100,
        render_sample=False
    )

    print("\n" + "=" * 40)
    print(f"🚀 模型评估报告（基于100次测试）")
    print("=" * 40)
    print(f"平均奖励: {metrics['average_reward']:.2f} ± {metrics['reward_std']:.2f}")
    print(f"成功率: {metrics['success_rate']:.2%}")
    print(f"每步平均燃料消耗: {metrics['fuel_per_step']:.4f}")
    print(f"可视化结果已保存: evaluation_plot.png")
    print("=" * 40)
