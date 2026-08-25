import time
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch
from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
torch.set_num_threads(1)

env = make_vec_env("HalfCheetah-v4", n_envs=int(os.environ.get("N_ENVS", "4")), seed=0)
total_timesteps = int(os.environ.get("TRAIN_STEPS", "100000"))

# 随机策略基线
baseline_env = gym.make("HalfCheetah-v4")
obs, _ = baseline_env.reset()
r0 = 0
for _ in range(1000):
    a = baseline_env.action_space.sample()
    obs, r, term, trunc, _ = baseline_env.step(a)
    r0 += r
    if term or trunc:
        break
baseline_env.close()

model = SAC("MlpPolicy", env, device="cuda", learning_rate=3e-4,
            batch_size=512, gradient_steps=2, verbose=0)
t0 = time.time()
model.learn(total_timesteps=total_timesteps)
dt = time.time() - t0
print(f"训练 {total_timesteps} 步耗时 {dt:.1f}s, 约 {total_timesteps/dt:.0f} 步/秒")
print("GPU 显存峰值:", round(torch.cuda.max_memory_allocated()/1e6), "MB")
env.close()

# 训练后评估
eval_env = gym.make("HalfCheetah-v4")
obs, _ = eval_env.reset()
r1 = 0
for _ in range(1000):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = eval_env.step(a)
    r1 += r
    if term or trunc:
        break
eval_env.close()
print(f"随机策略 reward:  {r0:.0f}")
print(f"训练后 reward:    {r1:.0f}")

##新增内容##
from gymnasium.wrappers import RecordVideo

video_env = gym.make(
    "HalfCheetah-v4",
    render_mode="rgb_array"
)
video_env = RecordVideo(
    video_env,
    video_folder="/work/videos",
    episode_trigger=lambda episode_id: True
)

obs, _ = video_env.reset()
action_sum = 0.0
frame_count = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    action_sum += abs(action).mean()
    frame_count += 1
    obs, reward, terminated, truncated, _ = video_env.step(action)

    if terminated or truncated:
        break

video_env.close()
print(f"视频阶段平均关节动作幅度: {action_sum / frame_count:.3f}")
model.save("/work/models/halfcheetah_sac")
