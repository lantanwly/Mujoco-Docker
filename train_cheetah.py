import time
import torch
from stable_baselines3 import SAC
import gymnasium as gym

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
env = gym.make("HalfCheetah-v4")

# 随机策略基线
obs, _ = env.reset()
r0 = 0
for _ in range(1000):
    a = env.action_space.sample()
    obs, r, term, trunc, _ = env.step(a)
    r0 += r
    if term or trunc:
        break

model = SAC("MlpPolicy", env, device="cuda", learning_rate=3e-4, batch_size=256, verbose=0)
t0 = time.time()
model.learn(total_timesteps=100_000)
dt = time.time() - t0
print(f"训练 100k 步耗时 {dt:.1f}s, 约 {100000/dt:.0f} 步/秒")
print("GPU 显存峰值:", round(torch.cuda.max_memory_allocated()/1e6), "MB")

# 训练后评估
obs, _ = env.reset()
r1 = 0
for _ in range(1000):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    r1 += r
    if term or trunc:
        break
print(f"随机策略 reward:  {r0:.0f}")
print(f"训练后 reward:    {r1:.0f}")
