reward using predicted position
**reward = 0.4r1 + 0.6r2**
r1 = e^(-(20x)^2)*
r2 = e^(-(50x)^4)

state = obj_to_neck_xyz, obj_to_hand_xyz, joint_arm, actions
range of obj_to_hand_xyz changed to -0.02~0.02

**learning rate = 0.0015**

ctrlpos = ctrlpos + action
Increase new point spawn interval to 5 sec (was 3), to let agent learn staying still when obj isn't moving.

+-1 degree noise in joint observation state
range of vec_target2hand increase to +-0.05
add target elbow yaw angle in state
angle_incre = anngle_incre += action *0.1

**Hidden layer (Actor):  64x64**
**Hidden layer (Critic): 256x256**