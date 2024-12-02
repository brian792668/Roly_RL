reward = r0*r2 + r3
**r3 = e^(-(50x)^4)**

state = **obj_to_neck_xyz**, obj_to_hand_xyz, joint_arm
range of obj_to_neck_xyz's y changed to **0~-0.75** (was 0.6~-0.6)

learning rate = 0.0005

**ctrlpos = ctrlpos + action**
**Increase new point spawn interval to 5 sec (was 3), to let agent learn staying still when obj isn't moving.**