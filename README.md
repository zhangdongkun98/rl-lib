# rl-lib

Goal: follow the structure of [openai-spinningup-key-papers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html).


## Requirements

### 0. some packages

```bash
pip install numpy==1.20.0
```

### 1. torch
```bash
pip install torch==1.7.1
```

### 2. gym
```bash
pip install gym
pip install box2d-py
pip install atari-py
```

### 3. MuJoCo and mujoco-py

[MuJoCo download](https://mujoco.org/download) <br>
[mujoco-py repo](https://github.com/openai/mujoco-py) <br>

- version 200
```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://roboti.us/download/mujoco200_linux.zip  ## extract and rename to mujoco200
wget https://roboti.us/file/mjkey.txt
./mujoco200/bin/simulate ./mujoco200/model/humanoid.xml  ## test if success
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin" >> ~/.bashrc

sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf

sudo apt-get install libglew-dev
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" >> ~/.bashrc

pip install mujoco-py==2.0.2.7
```

- version 210 (todo)
```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
```



## Installation

```bash
git clone https://github.com/zhangdongkun98/rl-lib.git
cd rl-lib
pip install -e .
```




## others
	- basic

    - buffer
    - template



## reference

- https://github.com/sfujim/TD3
- https://github.com/haarnoja/sac
- https://github.com/pranz24/pytorch-soft-actor-critic

