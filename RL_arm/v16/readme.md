## PPO & SAC
* 50Hz
* **5 DOF, 上臂彎**
* action: 角度變化
* obs: 無hand2target
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward:
* r0: reward of position
* r1: panalty of leaving
* r2: reward of handCAM central
* r3: reward of detail control
* reward = r0 * (1+0.5*r2) + r1 + r3