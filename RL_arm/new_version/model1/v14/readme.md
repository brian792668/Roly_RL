reward = r0*r2 + r3
**r3 = e^(-(50x)^4)**

state = obj_to_hand_xyz, joint_arm

learning rate = 0.0005

**ctrlpos = ctrlpos + action**
**Increase new point spawn interval to 5 sec (was 3), to let agent learn staying still when obj isn't moving.**