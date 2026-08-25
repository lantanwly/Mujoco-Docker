FROM torch-test
# RL 栈:SB3 + gymnasium(锁兼容版本)+ numpy
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "stable-baselines3==2.4.1" "gymnasium==0.29.1" numpy
