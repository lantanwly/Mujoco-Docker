# Docker + GPU 运行 RL / VLA 使用指南

> 针对这台机器(云 VM, Ubuntu 20.04, NVIDIA L20-12Q 12GB)已配好的 Docker GPU 环境。
> 包含:Docker 零基础入门、怎么用、原理是什么、踩过的坑和排查方法。

---

## 目录

1. [总览](#1-总览)
2. [Docker 入门(零基础)](#2-docker-入门零基础)
3. [机器环境速览](#3-机器环境速览)
4. [原理:GPU、CUDA、RL、瓶颈](#4-原理gpu-cuda-rl-瓶颈)
5. [使用方法](#5-使用方法)
6. [常见问题排查](#6-常见问题排查)
7. [下一步方向](#7-下一步方向)

---

## 1. 总览

在这台机器上,我们用 Docker 容器跑强化学习(RL)和 VLA。整体结构:

```
宿主机 (Linux + NVIDIA 驱动 535)
  └── Docker daemon (装了 nvidia-container-toolkit,注册了 nvidia runtime)
        └── 容器 (torch-test → rl-test → rl-mujoco,层层叠加)
              ├── Python 3.8
              ├── PyTorch 2.3.1+cu121  (GPU 计算)
              ├── stable-baselines3     (RL 算法库)
              ├── gymnasium             (环境接口)
              └── mujoco                (物理引擎)
```

三个镜像是逐层继承的:

| 镜像 | 内容 | 已验证 |
|---|---|---|
| `torch-test` | nvidia/cuda:12.2 base + Python + torch 2.3.1+cu121 | GPU 矩阵乘 ✅ |
| `rl-test` | + SB3 2.4.1 + gymnasium 0.29.1 | PPO 训练 CartPole ✅ |
| `rl-mujoco` | + mujoco 2.3.7 + GL 库 | SAC 训练 HalfCheetah ✅ |

如果你还不会 Docker,先看第 2 节,从零教你。

---

## 2. Docker 入门(零基础)

### 2.1 一句话理解 Docker

> Docker 把「程序 + 它的整个运行环境(系统库、Python 版本、所有依赖)」打包成一个独立的盒子,任何装了 Docker 的机器上都能原样跑起来,不受宿主机环境影响。

这台机器宿主机是 Python 3.8、没装 torch;但我们想要的 RL 环境在容器里,与宿主机互不干扰。这就是用 Docker 的意义。

### 2.2 最重要的心智模型:镜像 vs 容器

这两个词是 Docker 里最容易混的,先记住:

| | **镜像 (Image)** | **容器 (Container)** |
|---|---|---|
| 是什么 | 打包好的**模板**,只读 | 模板**运行起来的实例**,可写 |
| 类比 | 安装光盘 / 半成品食材包 | 装好的电脑 / 按菜谱做的一顿饭 |
| 类比(程序员版) | **类 (class)** | **对象 (object)** |
| 能改吗 | 不能直接改,只能重建 | 能改,但删了就没了 |

**要点**:一个镜像可以启动无数个容器,每个容器互相隔离、互不影响。

- `docker images` → 本机有哪些镜像(光盘)
- `docker ps` → 现在哪些容器在跑(电脑开着几台)
- `docker ps -a` → 所有容器,包括已退出的

### 2.3 常用命令速查(先背这几个就够用)

```bash
docker images                 # 查看本机镜像
docker pull <镜像名>          # 从仓库下载镜像,如 docker pull nvidia/cuda:12.2
docker run <镜像> <命令>      # 用镜像启动一个容器,执行命令
docker ps                     # 查看运行中的容器
docker ps -a                  # 查看所有容器(含已退出)
docker exec -it <容器> bash   # 进入一个运行中容器的 shell
docker stop <容器>            # 停止容器
docker rm <容器>              # 删除容器(释放占用的名字)
docker rmi <镜像>             # 删除镜像(释放磁盘)
docker system df              # 看磁盘被 Docker 占了多少
```

### 2.4 `docker run` 的常用参数(每个都配真实例子)

`docker run` 是你用得最多的命令,参数决定容器怎么跑:

| 参数 | 作用 | 本项目例子 |
|---|---|---|
| `--rm` | 容器退出后自动删除(一次性任务不残留) | 我们每次跑训练都加它 |
| `-it` | 交互模式 + 伪终端(让你能进 shell 打字) | `docker run -it rl-mujoco bash` |
| `--gpus all` | **把 GPU 给容器用**(不加就看不到 GPU!) | 所有训练命令都加 |
| `-v 宿主机目录:容器目录` | 挂载卷,两边共享同一份文件 | `-v /home/test/torch-test:/work` |
| `--name <名字>` | 给容器起名,方便 stop/rm | `docker run --name mytrain ...` |
| `-d` | 后台运行(输出不挡终端) | 长任务用 |

镜像名后面跟的**命令**,是「在这个容器里执行什么」:

```bash
# 启动容器 → 在容器里执行 python3,跑 /work 下的脚本
docker run --rm --gpus all -v $PWD:/work rl-mujoco python3 /work/train_cheetah.py
#            ▲     ▲        ▲            ▲          ▲
#          一次性  给GPU   挂载目录      哪个镜像   容器里执行的命令
```

### 2.5 容器里改的文件会丢!—— 卷 (volume) 的意义

容器里有个「错觉」:文件系统看起来正常,但**容器一删除,里面所有改动都没了**。

所以我们要用 `-v 宿主机目录:容器目录` 挂载卷:

```
宿主机 /home/test/torch-test/xxx.py  ⇄  容器 /work/xxx.py   (同一份文件)
```

两边看到的是**同一份文件**,任一方修改,另一方立即能看到。这样:

- 你的脚本写在宿主机(方便用编辑器),容器里直接跑
- 训练出的模型 `.zip` 保存在挂载目录里 → 落在宿主机,容器删了也不丢

**教训**:如果不挂载,训练 10 分钟存下的模型,容器一 `--rm` 就没了。

### 2.6 Dockerfile:怎么做一个属于自己的镜像

`docker run` 是在**别人的镜像**里跑命令。想定制环境(装库),就写 Dockerfile 再 build:

```dockerfile
FROM rl-mujoco                     # 基于哪个镜像(继承它的全部内容)
RUN pip3 install -i <清华源> gymnasium   # 构建时执行:装库
CMD ["python3"]                    # 容器启动时默认执行的命令
```

```bash
docker build -f Dockerfile.xxx -t 新名字 .   # 构建成新镜像
```

最常用的指令只有三个:
- `FROM` → 站在谁的肩上
- `RUN` → 构建时跑的命令(装包、配环境)
- `CMD` → 容器启动时默认跑什么(可以不写,`docker run` 时会覆盖)

`RUN` 每次都会生成一层,`build` 时如果前面没改,后面的层会复用缓存(所以加一层库很快)。

### 2.7 上手练习(按顺序做一遍,5 分钟)

**练习 1:跑第一个容器**

```bash
docker run --rm hello-world      # 输出一段欢迎语后自动退出(--rm 让它用完即删)
```

**练习 2:进容器看看里面有什么**

```bash
docker run --rm -it rl-mujoco bash     # 进入容器 shell
python3 --version                      # 看容器里的 Python(3.8,和宿主机一样,但环境干净)
torch  # 会报错,正常——python3 里 import torch 试试:
python3 -c "import torch; print(torch.__version__)"   # 能打印 2.3.1+cu121
exit                                  # 退出容器
```

**练习 3:挂载目录,验证共享**

```bash
cd /home/test/torch-test
echo "hello from host" > test.txt
docker run --rm -v $PWD:/work rl-mujoco cat /work/test.txt   # 容器能读到宿主机的文件
rm test.txt
```

**练习 4:自己 build 一个镜像**

```bash
# 照抄第 2.6 节的 Dockerfile,存成 Dockerfile.test
docker build -f Dockerfile.test -t mytest .
docker run --rm mytest   # 跑出你自己的镜像
```

### 2.8 常见认知误区(先打预防针)

| 你以为 | 实际 |
|---|---|
| 改了宿主机的文件,容器里也会变 | 只有**挂载卷**里的文件才会;镜像文件是只读快照 |
| 容器里能直接看到 GPU | 必须 `--gpus all`(第 2.4 节),否则「没驱动」 |
| 容器退出=环境没了 | 容器是临时的,环境保存在**镜像**里,可以反复启动 |
| 镜像越跑越大 | 镜像只读、大小固定;运行时的数据在容器里(临时)或卷里 |
| 重新 `docker run` 会接着上次的状态 | 不会,每次都是**全新启动**,状态不保留(要保留就存文件到卷) |

---

## 3. 机器环境速览

- **GPU**: NVIDIA L20-12Q, **12GB 显存**(云上虚拟化的 L20 切片;完整 L20 是 48GB)
- **驱动**: 535.161.07, 支持 CUDA 最高 **12.2**
- **CPU**: Intel Xeon Gold 6444Y × **8 核**
- **内存**: 62GB; 磁盘: 400GB 可用
- **网络**: 在中国区,Docker Hub 直连被限速(~600KB/s),**无 IPv6 路由**

**关键结论(显存决定一切)**:
- RL:完全能跑 ✅
- VLA 推理:小模型(3B 级)可以,OpenVLA 7B 只能量化
- VLA 训练/微调:12GB 不够,基本别想

---

## 4. 原理:GPU、CUDA、RL、瓶颈

### 4.1 Docker 怎么把 GPU 给容器

容器默认看不到 GPU。要打通,需要三件事:

1. **宿主机有 NVIDIA 驱动** —— 驱动分两部分:
   - 内核模块(`nvidia`,管理硬件)
   - 用户态库(`libcuda.so`,应用通过它调用驱动)
   容器共享宿主机内核,但默认访问不到这两样。

2. **装 nvidia-container-toolkit** —— 它提供一个 `nvidia-container-cli`,在容器启动时注入:
   - 设备文件:`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`
   - 驱动用户态库:`libcuda.so` 等(通过 `LD_LIBRARY_PATH` 暴露给容器)
   - 这样容器里的程序就能调用宿主机的 GPU 驱动了。

3. **注册 `nvidia` runtime 到 Docker** —— `nvidia-ctk runtime configure` 在 `/etc/docker/daemon.json` 里加:

   ```json
   { "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "args": [] } } }
   ```
   之后 `docker run --gpus all` 就会自动走这个 runtime,做上面的注入。

> **`--gpus all` 和 `--runtime nvidia` 等价**。没装 toolkit 时 `--gpus all` 会报
> `could not select device driver "" with capabilities: [[gpu]]`。
>
> **Docker 不占用宿主机驱动**:驱动是宿主机的资源,容器只是「借用」它的视图,多个容器可以同时共享;容器退出后借走的东西自动归还。

### 4.2 为什么 CUDA 版本必须匹配

这里有个容易混淆的「两个 CUDA」:

| 名词 | 在哪 | 干什么 |
|---|---|---|
| **驱动 (driver)** | 宿主机,内核态 | 提供 `libcuda.so`,`cu121/cu122` 这类标签里的 runtime 最终都调它 |
| **运行时库 (runtime)** | 容器镜像内 | `libcudart`、`libcublas`、`cuDNN` 等,是 torch 真正链接的库 |

我们 pip 装 torch 时,自动带上了 `nvidia-*-cu12` 这一堆 pip 包——它们就是容器里的 CUDA 运行时库。

**兼容规则(向前兼容)**:驱动 ≥ 运行时所需的最低版本即可。
- 驱动 535 ≈ CUDA 12.2
- torch 2.3.1 的 cu121 构建需要驱动 ≥ 525 → **能跑**
- cu124/cu126/cu128 构建需要驱动 ≥ 550/525/570 → 这台机器 **跑不了**

所以:这台机器上装 torch 必须用 **cu121**(或更旧的 cu118),否则会报
`CUDA driver version is insufficient for CUDA runtime version`。

### 4.3 强化学习基本原理

RL 解决的是**序贯决策**问题,建模为 MDP(马尔可夫决策过程):

```
         ┌───────────────────────────────────┐
         │           环境 Environment       │
         │   (MuJoCo 物理引擎 / 游戏 / 机器人) │
         └──────┬──────────────▲────────────┘
        action  │              │ state + reward
         ┌──────▼──────────────┴────────────┐
         │           智能体 Agent           │
         │   策略 π(a|s) + 价值函数 Q(s,a)   │
         │        (神经网络,跑在 GPU)         │
         └──────────────────────────────────┘
```

四个要素:
- **state s**: 当前状态(如机器人的关节角、角速度)
- **action a**: 智能体做出的动作(连续值 = 关节力矩)
- **reward r**: 环境给的反馈(如向前速度奖励)
- **策略 π(a|s)**: 智能体「看到状态 → 选动作」的函数,就是要学的东西

**目标**: 最大化累计(折扣)奖励 `Σ γᵗ rₜ`。

**训练循环**(三大步反复):
1. **采样**: 用当前策略跑环境,收集 (state, action, reward, next_state)
2. **更新**: 用这些数据更新神经网络(策略 + 价值函数)
3. 重复,直到策略收敛

### 4.4 PPO 和 SAC 是什么

我们用了两个算法,各自适用场景不同:

| | **PPO** | **SAC** |
|---|---|---|
| 全称 | Proximal Policy Optimization | Soft Actor-Critic |
| 类型 | on-policy(在线) | off-policy(离线) |
| 样本效率 | 低(每批数据用一次就丢) | 高(存回放缓冲区反复用) |
| 稳定性 | 很稳,常用默认选择 | 稳,超参敏感一点 |
| 动作空间 | 离散/连续 | **连续控制最佳** |
| 核心思想 | 每次只改一小步,用「裁剪的替代目标」限制策略变化幅度 | 最大化熵 + 最大化奖励,鼓励探索 |

所以:
- CartPole(离散动作)→ 我们用 PPO
- HalfCheetah(连续关节力矩)→ 我们用 SAC

**GPU 在 RL 里的角色**: 网络前向/反向传播(更新策略)用 GPU;环境采样(物理模拟)在 CPU。数据从 CPU 搬到 GPU 再搬回来,这个来回在小模型上开销不小。

### 4.5 MuJoCo 是什么

- MuJoCo (Multi-Joint dynamics with Contact) 是**刚体物理模拟器**,专门模拟机器人/动物的关节运动、接触、约束。
- 它是 **CPU 计算**、单环境单线程,所以训练 HalfCheetah 时瓶颈在 CPU 采样。
- gymnasium 把 MuJoCo 包成标准 Gym 环境(`HalfCheetah-v4`、`Ant-v4`、`Hopper-v4`…),接口统一为 `reset()/step(action)→(obs, reward, terminated, truncated, info)`。
- HalfCheetah 的 reward 主要奖励向前的速度,所以训练后 reward 从 -362 涨到 +4000+ 就代表它真的「跑起来了」。

### 4.6 向量化并行与瓶颈(我们实测的教训)

**为什么 8 个并行环境没让训练总时长缩短?**

1. `make_vec_env(n_envs=8)` 用 8 个子进程各跑一个 MuJoCo 环境 → 采样吞吐从 108 提到 829 步/秒(7.7×)。
2. 但 SB3 默认 `gradient_steps=1`:每采一轮(8 步)只做 1 次梯度更新。数据多了 8 倍、更新没跟上 → **欠拟合**,reward 掉到 20。
3. 把 `gradient_steps` 提到 8,恢复 1:1 更新密度 → reward 回到 4271,但墙钟时间又被「100k 次小梯度更新」占满了。

本质是 **Amdahl 定律**:采样和更新是流水线的两端,只优化其中一端,另一端变成瓶颈。

- 这台机器上,小 MLP 的梯度更新主要是 **kernel 启动延迟**(~ms 级,与批次大小关系不大)。
- **真正的提速杠杆是增大 `batch_size`**(如 1024)配小 `gradient_steps`(如 2):用更少次、更大的更新摊销启动开销。
- 如果你的网络很大(如 VLA 模型),GPU 算力才是瓶颈,那时向量化采样提速会很显著。

---

## 5. 使用方法

### 5.0 项目骨架(推荐直接用这个)

`/home/test/rl-project/` 是一个可直接用的 RL 项目骨架:

```
rl-project/
├── run.sh        # 统一启动入口:包装 docker run(带 --gpus all / -v / --user)
├── train.py      # 通用训练:--env --algo --timesteps --n-envs --save
├── eval.py       # 加载模型评估:--model --env --episodes
└── models/       # 训练产物(在挂载卷里,持久在宿主机)
```

```bash
cd /home/test/rl-project

# 训练(默认 HalfCheetah + SAC,8 并行环境,10 万步)
./run.sh train.py --save models/cheetah

# 换环境/算法/步数
./run.sh train.py --env Ant-v4 --algo ppo --timesteps 50000 --save models/ant_ppo

# 评估已保存的模型
./run.sh eval.py --model models/cheetah --env HalfCheetah-v4 --episodes 5
```

`run.sh` 等价于手写:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
       -v $PWD:/work -w /work rl-mujoco python3 <脚本> <参数>
```

### 5.1 直接跑现成的 demo

```bash
# 目录约定:脚本放在宿主机 /home/test/torch-test/,挂载进容器 /work
cd /home/test/torch-test

# 1) 验证 GPU 可用
docker run --rm --gpus all torch-test

# 2) CartPole + PPO (离散控制,几十秒)
docker run --rm --gpus all -v $PWD:/work rl-test \
    python3 /work/train_cartpole.py

# 3) HalfCheetah + SAC (连续控制,单环境 ~15 分钟)
docker run --rm --gpus all -v $PWD:/work rl-mujoco \
    python3 /work/train_cheetah.py

# 4) HalfCheetah + SAC (8 并行环境)
docker run --rm --gpus all -v $PWD:/work rl-mujoco \
    python3 /work/train_cheetah_vec.py
```

### 5.2 换环境、改参数

训练脚本就是普通 Python。常见改法:

```python
# 换环境:HalfCheetah → Ant / Hopper / Walker2d ...
env = gym.make("Ant-v4")

# 换算法:PPO / SAC / TD3 / DQN
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, device="cuda", ...)

# 关键超参
# - total_timesteps: 训练总步数(HalfCheetah 学得差不多要 30w+)
# - learning_rate:   SAC 通常 3e-4
# - batch_size:      256 起步;向量化时增大到 1024 + 降 gradient_steps
# - device="cuda":   用 GPU,注意容器里必须有 --gpus all
```

### 5.3 保存 / 加载模型

```python
model.save("/work/models/halfcheetah_sac")      # 挂载到 /work,落在宿主机

from stable_baselines3 import SAC
model = SAC.load("/work/models/halfcheetah_sac", env=env)
```

### 5.4 自己加库 / 新建镜像

三个镜像都可以直接跑 `pip install`(一次性,不持久):

```bash
docker run --rm -v $PWD:/work rl-mujoco pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
```

想做成新镜像,改 Dockerfile 再 build(参考 `Dockerfile.rl` / `Dockerfile.mujoco`):

```dockerfile
FROM rl-mujoco
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple <新库>
```

```bash
docker build -f Dockerfile.xxx -t 名字 .
```

> 国内网络必备:apt 和 pip 都走清华源;新装 CUDA 相关包要选 **cu121** 版本。

---

## 6. 常见问题排查

| 报错 | 原因 | 解决 |
|---|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | 没装/没注册 nvidia runtime | 装 nvidia-container-toolkit + `nvidia-ctk runtime configure` + 重启 docker |
| 容器里 `nvidia-smi` 说「没有驱动」 | 容器没加 `--gpus all` | 加 `--gpus all`(宿主机驱动一直是好的) |
| `dial tcp [IPv6...]: connect: network is unreachable` | 机器无 IPv6 路由,Docker 解析到 AAAA 记录 | `/etc/gai.conf` 加 `precedence ::ffff:0:0/96 100`,重启 docker |
| `CUDA driver version is insufficient for CUDA runtime version` | torch 构建的 CUDA 比驱动新 | 用 cu121 构建(驱动 535 支持到 12.2) |
| `ModuleNotFoundError: No module named 'imageio'` | gymnasium 渲染器无条件 import | 装 imageio + imageio-ffmpeg |
| MuJoCo env 创建时 GL 相关报错 | 无头容器缺图形库 | apt 装 libgl1 libosmesa6 libglfw3 等(已在 rl-mujoco 里装好) |
| SB3 装完报 gymnasium 版本冲突 | SB3 2.4.x 不支持 gymnasium 1.1+ | 锁 `gymnasium==0.29.1` |
| 向量化后 reward 骤降 | 数据多了但梯度更新没跟上 | 同步放大 `gradient_steps` |
| 重新 `docker run` 后模型不见了 | 没挂载目录,容器是全新的 | 用 `-v` 把模型目录挂载出来 |

---

## 7. 下一步方向

- **Isaac Lab 具身 RL**:真实机器人框架(四足、机械臂),需要 Python 3.10+ 的新镜像,12GB 显存可跑小型任务
- **π0 VLA 推理**:3B 视觉-语言-动作模型,12GB 显存下可做推理
- **OpenVLA 微调**:需要 48GB 显存,这台机器做不了

相关文件都在 `/home/test/torch-test/`:
- `Dockerfile`(torch-test)、`Dockerfile.rl`、`Dockerfile.mujoco`
- `GUIDE.md`(本指南)
- `train_cartpole.py`、`train_cheetah.py`、`train_cheetah_vec.py`
