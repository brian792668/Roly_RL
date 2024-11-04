## PPO & SAC
* 50Hz
* 4 DOF, 上臂直
* action: 角度變化
* obs: 無hand2target
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward: **新增r3**
 * r0: reward of position
 * r1: panalty of leaving
 * r2: reward of handCAM central
 * **r3: reward of detail control**
 * reward = r0 * (1+0.5*r2) + r1 + r3