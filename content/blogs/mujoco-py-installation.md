---
title: "Configuring the mujoco-py Environment for Reinforcement Learning"
date: 2024-01-03T22:35:11+08:00
draft: false
author: Wenqian
tags: [mujoco-py, rl-env]
image:
description:
toc: true
---

## Preface

Sometimes it's necessary to set up an reinforcement learning environment, but I found that I have to start the configuration from scratch every time, so it's better to keep a record...

## Quick Installation of Mujoco Environment

- Step 1: Download to the `/root/` directory (Terminal's `~/` directory)
```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
```

- Step 2: Extract to `.mujoco/`
```bash
mkdir ~/.mujoco # Create
tar -zxvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

- Step 3: Append environment information
```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.bashrc 
source .bashrc
```

- Step 4: Install relevant Python libraries in the corresponding conda environment
```bash
pip install mujoco
pip install mujoco-py
```

---
Changing sources: Ubuntu's software source configuration file is `/etc/apt/sources.list`. Make a backup of the system's default file and replace it with the following content to use the selected software source mirror. (Although it may not be necessary on a PC, it may be necessary on a server)

```bash
# The source code mirror is commented out by default to improve apt update speed. Uncomment if needed
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse

# Pre-release software source, not recommended for enable
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```

You can use the vim command to complete it:
```bash
vim /etc/apt/sources.list
### --- ###
# In vim view
# Clear sources
# `gg` jump to the beginning of the file, d delete G until the end of the file
ggdG
# Then paste the above code
```

Dependency installation
```bash
apt-get update
apt install libosmesa6-dev libgl1-mesa-glx libglfw3
apt-get install patchelf
```

Note: Some other installations may be required, refer directly to the [github](https://github.com/openai/mujoco-py) repository of mujoco-py.