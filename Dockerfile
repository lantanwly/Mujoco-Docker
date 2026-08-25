FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# apt 走清华源,加快依赖安装
RUN sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g; s@http://security.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# pip 走清华源装 torch 2.3.1 (cu121,适配驱动 535/CUDA 12.2)
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.3.1

CMD ["python3", "-c", "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"]
