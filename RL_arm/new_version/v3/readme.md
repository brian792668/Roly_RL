## Model 1
* 50Hz
* 3 DOF, 上臂直
* action: 角度變化
* obs: **相對於neck的xyz**
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward:
* r0: reward of position
* r1: panalty of leaving
* r2: reward of handCAM central
* r3: reward of detail control
* reward = r0 * r2 + r1 + r3
**使arm1隨機生成角度接近身體**

## Model 2
* 50Hz
* 1 DOF, 上臂直
* action: 角度變化
* obs: 相對於neck的xyz
* hand camera更新為虛擬，並放置在手掌中央以改善偏移
### reward:
* reward = **- sum of torque**