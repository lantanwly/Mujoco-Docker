import time
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))

# 8 个并行环境 (SubprocVecEnv),吃满 8 核 CPU
env = make_vec_env("HalfCheetah-v4", n_envs=8, seed=0)

model = SAC("MlpPolicy", env, device="cuda", learning_rate=3e-4, batch_size=256, gradient_steps=8, verbose=0)
t0 = time.time()
model.learn(total_timesteps=100_000)
dt = time.time() - t0
print(f"训练 100k 步耗时 {dt:.1f}s, 约 {100000/dt:.0f} 步/秒")
print("GPU 显存峰值:", round(torch.cuda.max_memory_allocated()/1e6), "MB")

# 单环境评估
eval_env = gym.make("HalfCheetah-v4")
obs, _ = eval_env.reset()
r1 = 0
for _ in range(1000):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = eval_env.step(a)
    r1 += r
    if term or trunc:
        break
print(f"训练后 reward: {r1:.0f}")
