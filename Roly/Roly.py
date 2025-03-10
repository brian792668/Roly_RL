import sys
import os
import threading
import numpy as np
import time
from stable_baselines3 import SAC  
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports.Camera import *
from imports.Forward_kinematics import *
from imports.Roly_motor import *

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(64, 128) # 隱藏層64輸出128
        self.fc3 = nn.Linear(128, 4)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第1層用 ReLU
        x = F.relu(self.fc2(x))  # 第2層用 ReLU
        x = self.fc3(x)          # 最後一層直接輸出
        return x

class Robot_system:
    def __init__(self):
        print("\033[1;33m[ Status ]\033[0m Initializing system ...")
        # Initial system
        self.system_running = True
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.time_start = time.time()

        # Initial camera
        self.head_camera = Camera()
        self.target_exist = self.head_camera.target_exist
        self.target_depth = self.head_camera.target_depth
        self.target_pixel_norm = self.head_camera.target_norm
        self.track_done = False

        # Initial RL policy
        RL_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RLmodel/model_1/v23-future2/model.zip")
        self.RL_moving_policy = SAC.load(RL_path1)
        self.RL_moving_action = [0] * 3
        RL_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RLmodel/model_2/v21-18/model.zip")
        self.RL_grasping_policy = SAC.load(RL_path2)
        self.RL_grasping_action = [0] * 3 
        self.grasping_dis = 0.1    
        
        # Initial natural pose model
        self.IK = IKMLP()
        self.IK.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "IKmodel_v7.pth"), weights_only=True))
        self.IK.eval()

        # Initial motors
        self.motor = Roly_motor()
        self.motor.to_pose(pose="initial", speed=0.5)

        # Initial mechanism
        self.DH_table_R = DHtable([[    0.0, np.pi/2,   -0.028,  -0.104],
                                   [np.pi/2, np.pi/2,     0.0,  0.2488],
                                   [    0.0,     0.0, -0.1105,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.1403],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1803]])
        self.DH_table_L = DHtable([[    0.0, np.pi/2,   -0.028,  -0.104],
                                   [np.pi/2, np.pi/2,     0.0, -0.2488],
                                   [    0.0,     0.0, -0.1105,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.1403],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1803]])

        self.pos_hand     = self.DH_table_R.forward(angles=np.radians(self.motor.joints[2:7].copy()))
        self.pos_target   = self.pos_hand.copy()
        self.pos_guide    = self.pos_hand.copy()
        self.pos_grasppnt = [ 0.0, 0.0, 0.0]
        self.pos_placepnt = [ 0.35, -0.1000, -0.150]
        self.pos_initpnt  = [ 0.15, -0.2488, -0.350]
        self.pos_shoulder = [-0.02, -0.2488, -0.104]
        self.pos_elbow    = [-0.02, -0.2488, -0.350]
        
        # Initial Status
        self.status = "waiting" # grasping, carrying
        self.grasping = False

    def thread_camera(self):
        self.head_camera.start()
        while not self.stop_event.is_set():
            # 50 Hz
            time.sleep(0.01)

            self.head_camera.get_img(rgb=True, depth=True)
            self.head_camera.get_target(depth=True)
            self.head_camera.get_hand()
            self.head_camera.show(rgb=True, depth=False)

            if self.head_camera.target_exist == True:
                with self.lock:
                    self.target_exist = True
                    self.track_done = False
                    self.target_pixel_norm = self.head_camera.target_norm
                    self.motor.joints_increment[0] = -3.0*self.head_camera.target_norm[0] - 0.5*self.head_camera.target_vel[0] - 0.7*self.motor.joints_increment[0]
                    self.motor.joints_increment[1] = -3.0*self.head_camera.target_norm[1] - 0.5*self.head_camera.target_vel[0] - 0.7*self.motor.joints_increment[1]
                if np.abs(self.head_camera.target_norm[0]) <= 0.05 and np.abs(self.head_camera.target_norm[1]) <= 0.05 :
                    with self.lock:
                        self.target_depth = self.head_camera.target_depth
                        self.track_done = True
            else:
                with self.lock:
                    self.target_exist = False
                    self.track_done = False
                    self.motor.joints_increment[0] = 0
                    self.motor.joints_increment[1] = 0
        
        self.head_camera.stop()

    def thread_system(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)

            # Farward Kinematics of EE position
            with self.lock: 
                angles=self.motor.joints.copy()
            angles = [ np.radians(angles[i]) for i in range(len(angles))]
            pos_hand = self.DH_table_R.forward(angles=angles[2:7].copy())
            with self.lock:
                self.pos_hand = pos_hand.copy()
                
            # Farward Kinematics of target position
            neck_angle = angles[0]
            camera_angle = angles[1]
            with self.lock: 
                track_done = self.track_done
                d = self.target_depth + 0.02
            if track_done:
                a = 0.062
                b = (a**2 + d**2)**0.5
                beta = np.arctan2(d, a)
                gamma = np.pi/2 + camera_angle-beta
                d2 = b*np.cos(gamma)
                grasp_point = [d2*np.cos(neck_angle), d2*np.sin(neck_angle), b*np.sin(gamma)]
                # if self.reachable(grasp_point.copy()) == True: 
                #     with self.lock:
                #         self.pos_grasppnt = grasp_point.copy()
                with self.lock:
                    self.pos_grasppnt = grasp_point.copy()

            # Update status
            with self.lock:
                status = self.status
                pos_grasppnt = self.pos_grasppnt.copy()
            reachable = self.reachable(pos_grasppnt.copy())
            if status == "waiting":
                if track_done and reachable:
                    with self.lock:
                        self.status = "grasping"
                        self.pos_target = pos_grasppnt.copy()
            elif status == "grasping":
                if track_done and reachable:
                    with self.lock:
                        self.pos_target = pos_grasppnt.copy()
                    if angles[8] >= np.radians(94.0):
                        with self.lock:
                            self.status = "carrying"
                            self.pos_target = self.pos_placepnt.copy()
                else:
                    with self.lock:
                        self.status = "waiting"
                        self.pos_target = self.pos_initpnt.copy()
            elif status == "carrying":
                if angles[8] <= np.radians(1.0):
                    with self.lock:
                        self.status = "waiting"
                        self.pos_target = self.pos_initpnt.copy()

                # with self.lock:
                #     pos_grasppnt = self.pos_grasppnt.copy()
                #     pos_hand = self.pos_hand.copy()
                # dis2hand = ( (pos_grasppnt[0]-pos_hand[0])**2 + (pos_grasppnt[1]-pos_hand[1])**2 + (pos_grasppnt[2]-pos_hand[2])**2 ) **0.5
                # if dis2hand >= 0.10:
                #     with self.lock:
                #         self.status = "grasping"

            sys.stdout.write(f"\rstatus: {status}")
            sys.stdout.flush()

    def thread_RL_move(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)
            with self.lock:
                joints = self.motor.joints.copy()
                joints_increment = self.motor.joints_increment.copy()
                target_xyz = self.pos_target.copy()
                hand_xyz = self.pos_hand.copy()
                guide_xyz = self.pos_guide.copy()
                action_old = self.RL_moving_action.copy()

            # RL model1: shoulder + elbow pitch
            guidetohand = [guide_xyz[0]-hand_xyz[0], guide_xyz[1]-hand_xyz[1], guide_xyz[2]-hand_xyz[2]]
            distoguide = (guidetohand[0]**2 + guidetohand[1]**2 + guidetohand[2]**2) ** 0.5
            guidetohand_norm = guidetohand.copy()
            if distoguide >= 0.02:
                guidetohand_norm = [guidetohand[0]/distoguide*0.02, guidetohand[1]/distoguide*0.02, guidetohand[2]/distoguide*0.02]
            
            joints = [ np.radians(joints[i]) for i in range(len(joints))]
            state = np.concatenate([guide_xyz.copy(), guidetohand_norm.copy(), action_old[0:3], joints[2:4], joints[5:7]]).astype(np.float32)
            action, _ = self.RL_moving_policy.predict(state)
            action_new = [action_old[0]*0.9 + action[0]*0.1,
                          action_old[1]*0.9 + action[1]*0.1,
                          action_old[2]*0.8 + action[2]*0.2]   
                 
            alpha = 1-0.8*np.exp(-100*distoguide**2)
            joints_increment[2] = np.degrees( action_new[0]* 0.02*alpha ) # shoulder pitch
            joints_increment[3] = np.degrees( action_new[1]* 0.02*alpha ) # shoulder roll
            joints_increment[6] = np.degrees( action_new[2]* 0.02*alpha ) # elbow pitch
            
            # # elbow yaw
            # with torch.no_grad():  # 不需要梯度計算，因為只做推論
            #     desire_joints = self.IK(torch.tensor(guide_xyz.copy(), dtype=torch.float32)).tolist()
            # desire_joints[2] += 40
            # desire_joints = np.radians(desire_joints)
            # joints_increment[5] = np.degrees( ( joints[5]*0.9 + desire_joints[2]*0.1 ) - joints[5] ) # elbow yaw

            # desire_joints[3] = IK_elbow_pitch(guide_xyz.copy())
            # joints_increment[6] = np.degrees( ( joints[6]*0.9 + desire_joints[3]*0.1 ) - joints[6] ) # elbow yaw

            with self.lock:
                self.RL_moving_action = action_new.copy()
                self.motor.joints_increment = joints_increment.copy()
                # self.pos_guide = guide_xyz.copy()

    def thread_RL_grasp(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.02)
            with self.lock:
                joints = self.motor.joints.copy()
                joints_increment = self.motor.joints_increment.copy()
                target_xyz = self.pos_target.copy()
                hand_xyz = self.pos_hand.copy()
                guide_xyz = self.pos_guide.copy()
                action_old = self.RL_grasping_action.copy()
                grasp_dis = self.grasping_dis
                track_done = self.track_done

            if track_done:
                # calculate new guide
                new_guide = [0.0, 0.0, 0.0]
                new_guide[0] = target_xyz[0] - grasp_dis*np.cos(np.pi/2*action_old[1])*np.cos(np.pi/2*action_old[0])
                new_guide[1] = target_xyz[1] - grasp_dis*np.cos(np.pi/2*action_old[1])*np.sin(np.pi/2*action_old[0])
                new_guide[2] = target_xyz[2] + grasp_dis*np.sin(np.pi/2*action_old[1])  

                # get RL states
                target2guide = [target_xyz[0] - guide_xyz[0], target_xyz[1] - guide_xyz[1], target_xyz[2] - guide_xyz[2]]
                joints = [ np.radians(joints[i]) for i in range(len(joints))]
                state = np.concatenate([target_xyz.copy(), target2guide.copy(), [joints[5]], [action_old[2]]]).astype(np.float32)
                action, _ = self.RL_grasping_policy.predict(state)
                action_new = [action_old[0]*0.9 + action[0]*0.1,
                              action_old[1]*0.9 + action[1]*0.1,
                              action_old[2]*0.9 + action[2]*0.1]
                joints_increment[5] = np.degrees( action_new[2]* 0.05 ) # elbow yaw

                # calculate new grasp distance
                target2hand = [target_xyz[0] - hand_xyz[0], target_xyz[1] - hand_xyz[1], target_xyz[2] - hand_xyz[2]]
                dis2target = ( target2hand[0]**2 + target2hand[1]**2 + target2hand[2]**2 ) **0.5
                if dis2target <= 0.11:
                    grasp_dis *= 0.1
                else:
                    grasp_dis = 0.1

                # calculate grasping increment
                with self.lock:
                    status = self.status
                joints_increment[8] = 0
                if status == "grasping":
                    if dis2target <= 0.03:
                        joints_increment[8] = (np.radians(95)*0.98 + joints[8]*0.02) - joints[8]
                    else:
                        joints_increment[8] = (0*0.9 + joints[8]*0.10) - joints[8]
                elif status == "carrying":
                    if dis2target <= 0.03:
                        joints_increment[8] = (0*0.98 + joints[8]*0.02) - joints[8]
                    else:
                        joints_increment[8] = (np.radians(95)*0.9 + joints[8]*0.1) - joints[8]

                
                with self.lock:
                    self.pos_guide = new_guide.copy()
                    self.grasping_dis = grasp_dis
                    self.motor.joints_increment[5] = joints_increment[5]
                    self.motor.joints_increment[8] = joints_increment[8]

    def thread_motor(self):
        while not self.stop_event.is_set():
            time.sleep(0.01) # 100 Hz
            with self.lock: 
                joints = self.motor.joints.copy()
                joints_increment = self.motor.joints_increment.copy()
            for i in range(len(joints)):
                joints[i] += joints_increment[i]
            with self.lock: 
                self.motor.joints = joints.copy()
            self.motor.writeAllMotorPosition(self.motor.toRolyctrl(joints.copy()))
            
        self.motor.to_pose(pose="shut down", speed=0.3)
        self.motor.setAllMotorTorqurDisable()
        self.motor.portHandler.closePort()

    def run(self, endtime = 10):
        print("\033[1;33m[ Status ]\033[0m Started.")
        threads = [ threading.Thread(target=self.thread_camera),
                    threading.Thread(target=self.thread_motor),
                    threading.Thread(target=self.thread_system),
                    threading.Thread(target=self.thread_RL_move),
                    threading.Thread(target=self.thread_RL_grasp) ]
        for t in threads:
            t.start()

        time.sleep(4)
        time0 = self.time_start
        while not self.stop_event.is_set():
            time.sleep(0.1)
            if time.time() - time0 >= endtime:  # 執行 10 秒後結束
                self.stop_event.set()

        for t in threads:
            t.join()

        cv2.destroyAllWindows()
        print("\033[1;33m[ Status ]\033[0m Stopped.")

    def reachable(self, point):
        with self.lock:
            pos_shoulder = self.pos_shoulder.copy()
        point_xyz = point.copy()
        dis2shoulder = ( (point_xyz[0]-pos_shoulder[0])**2 + (point_xyz[1]-pos_shoulder[1])**2 + (point_xyz[2]-pos_shoulder[2])**2 ) **0.5
        if dis2shoulder >= 0.45 or dis2shoulder <= 0.25:
            return False
        else:
            return True

if __name__ == "__main__":
    Roly = Robot_system()
    Roly.run(endtime=40)
