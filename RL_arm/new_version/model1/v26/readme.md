**reward using predicted position**
reward = r0*r2 + r3
r0 = 0.8*e^(-(20x)^2)*
r3 = e^(-(50x)^4)

state = obj_to_neck_xyz, obj_to_hand_xyz, joint_arm, actions
range of obj_to_hand_xyz changed to -0.02~0.02

learning rate = 0.0005

ctrlpos = ctrlpos + action
Increase new point spawn interval to 5 sec (was 3), to let agent learn staying still when obj isn't moving.

**+-1 degree noise in joint observation state**
**range of vec_target2hand increase to +-0.02**
**change state space of target xyz**
**change reward function: reward of detail control**
