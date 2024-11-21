## Model 1
* 100Hz
* 3 DOF, 上臂直
* action: 角度變化
* obs: **相對於neck的xyz**
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward:
* r0: reward of position
* r1: panalty of leaving
* r2: reward of handCAM central
* r3: reward of detail control
* reward = 0.1*r0r2 + r1 + r3
**add reward to r1**

## Model 2
* 100Hz
* 1 DOF, 上臂直
* action: 角度變化
* obs: 相對於neck的xyz
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward:
* reward = **IK MSE**