## PPO & SAC
* 50Hz
* 5 DOF, 上臂彎, **R1R2獨立**
* action: 角度變化
* obs: 無hand2target
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward:
* r0: reward of position  **updata to e^(-30(x^1.8))**
* r1: panalty of leaving
* r2: reward of handCAM central
* r3: reward of detail control **updata to e^(-(20x)^2)**
* reward = r0 * (1+0.5*r2) + r1 + r3