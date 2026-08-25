import time
import torch
from stable_baselines3 import PPO
import gymnasium as gym

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, device="cuda", n_steps=256, batch_size=64,
            learning_rate=3e-4, verbose=0)

t0 = time.time()
model.learn(total_timesteps=20000)
dt = time.time() - t0
print(f"训练完成 20000 步,耗时 {dt:.1f}s,约 {20000/dt:.0f} 步/秒")
print("GPU 显存峰值:", round(torch.cuda.max_memory_allocated()/1e6, 1), "MB")

obs, _ = env.reset()
total = 0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, terminated, truncated, _ = env.step(action)
    total += r
    if terminated or truncated:
        break
print("评估:单个回合存活步数 =", int(total))
