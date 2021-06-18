

# LunarLanderContinuous-v2, TD3

- 令人震惊的发现... 激活函数为relu，橙色曲线（网络宽度为256），蓝色曲线（网络宽度为64）：
<p align="center"><img src="./images/2021-06-18-1.png" width="90%"><br></p>

---

- 网络宽度为256，蓝色曲线（激活函数为relu），红色曲线（激活函数为tanh）：
<p align="center"><img src="./images/2021-06-18-2.png" width="90%"><br></p>

```python
class Actor(Model):
    def __init__(self, config):
        super(Actor, self).__init__(config, model_id=0)

        self.fc = nn.Sequential(
            nn.Linear(config.dim_state, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, config.dim_action), nn.Tanh(),
        )
        self.apply(init_weights)
    
    def forward(self, state):
        return self.fc(state)


class Critic(Model):
    def __init__(self, config):
        super(Critic, self).__init__(config, model_id=0)

        self.fc1 = nn.Sequential(
            nn.Linear(config.dim_state+config.dim_action, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.fc2 = copy.deepcopy(self.fc1)
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc1(x), self.fc2(x)
    
    def q1(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc1(x)
```

---

# notes

- PPO的 K_epochs 参数不能过大，否则会越练越差(K_epochs=4 与 K_epochs=40的对比)

