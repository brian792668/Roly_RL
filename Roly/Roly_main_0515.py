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
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.tanh(x) * 95
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
        self.target_position_to_camera = [0, 0, 0]

        # Initial RL policy
        RL_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/RL_model_1/v28/model.zip")
        self.RL_moving_policy = SAC.load(RL_path1)
        self.RL_moving_action = [0] * 3
        RL_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/RL_model_2/v21-20_new/model.zip")
        self.RL_grasping_policy = SAC.load(RL_path2)
        self.RL_grasping_action = [0] * 3 
        
        # Initial natural pose model
        self.NP = NPMLP()
        self.NP.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/NP/model_v13.pth"), weights_only=True))
        self.NP.eval()
        self.NP_rate = 0.5
 
        # Initial motors
        self.motor = Roly_motor()
        self.motor.to_pose(pose="initial", speed=0.5)


        # Initial mechanism
        self.DH_table_R = DHtable([[    0.0, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,  0.2488],
                                   [    0.0,     0.0, -0.1705,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.2003],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, 0.0,     0.0,  0.1700]])
        self.DH_table_L = DHtable([[    0.0, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.2488],
                                   [    0.0,     0.0, -0.1705,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.2003],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1700]])
        self.DH_table_neck = DHtable([[     0.0,     0.0,   0.021,   0.104],
                                      [     0.0, np.pi/2,     0.0,     0.0],
                                      [ np.pi/2, np.pi/2,   0.062,     0.0]])

        self.grasping_dis = 0.0
        self.pos_hand     = self.DH_table_R.forward_hand(angles=np.radians(self.motor.joints[2:7].copy()), hand_length=self.grasping_dis)
        self.pos_target   = self.pos_hand.copy()
        self.pos_guide    = self.pos_hand.copy()
        self.pos_grasppnt = [ 0.000, -0.2488,  0.000]
        self.pos_placepnt = [ 0.000, -0.2488,  0.000]
        self.pos_initpnt  = [ 0.096, -0.3851, -0.375]
        self.pos_shoulder = [ 0.000, -0.2488, -0.104]
        self.pos_elbow    = [ 0.000, -0.2488, -0.350]
        
        # Initial Status
        self.status = "wait_to_grasp" # wait_to_grasp -> grasping -> carrying -> placing
        self.grasping = False
    
    def thread_status(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.1)

            # Update status
            with self.lock:
                status = self.status


            if status == "wait_to_grasp":
                with self.lock:
                    track_done = self.track_done
                    pos_grasppnt = self.pos_grasppnt.copy()
                reachable = self.reachable(pos_grasppnt.copy())
                if track_done and reachable:
                    with self.lock:
                        self.status = "move_to_grasp"
                        self.grasping_dis = 0.15
                        self.pos_target = pos_grasppnt.copy()
                        self.pos_guide = self.pos_target.copy()
            
            elif status == "move_to_grasp":
                with self.lock:
                    track_done = self.track_done
                    pos_grasppnt = self.pos_grasppnt.copy()
                    target_xyz = self.pos_target.copy()
                    hand_xyz = self.pos_hand.copy()
                reachable = self.reachable(pos_grasppnt.copy())
                if track_done and reachable:
                    with self.lock:
                        self.pos_target = pos_grasppnt.copy()
                        self.pos_guide = self.pos_target.copy()
                target2hand = [target_xyz[0] - hand_xyz[0], target_xyz[1] - hand_xyz[1], target_xyz[2] - hand_xyz[2]]
                dis_target2hand = ( target2hand[0]**2 + target2hand[1]**2 + target2hand[2]**2 ) **0.5
                with self.lock:
                    self.grasping_dis = 0.05 + 0.1*np.tanh(200*dis_target2hand)
                if dis_target2hand <= 0.03:
                    with self.lock:
                        self.grasping_dis = 0.05
                        self.status = "grasping"

            elif status == "grasping":
                with self.lock:
                    joints = self.motor.joints.copy()
                    joints_increment = self.motor.joints_increment.copy()
                joints = [ np.radians(joints[i]) for i in range(len(joints))]
                joints_increment[8] = (np.radians(85)*0.98 + joints[8]*0.02) - joints[8]
                with self.lock:
                    self.motor.joints_increment[8] = joints_increment[8]
                if joints[8] >= np.radians(84.8):
                    with self.lock:
                        self.status = "carrying"

            elif status == "carrying":
                with self.lock:
                    track_done = self.track_done
                    joints = self.motor.joints.copy()
                    joints_increment = self.motor.joints_increment.copy()
                    pos_placepnt = self.pos_placepnt.copy()
                pos_placepnt[2] += 0.10
                reachable = self.reachable(pos_placepnt.copy())
                if track_done and reachable:
                    pos_target = pos_placepnt.copy()
                    with self.lock:
                        self.status = "move_to_place"
                        self.grasping_dis = 0.15
                        self.pos_target = pos_target.copy()
                        self.pos_guide = self.pos_target.copy()

            elif status == "move_to_place":
                with self.lock:
                    target_xyz = self.pos_target.copy()
                    hand_xyz = self.pos_hand.copy()
                target2hand = [target_xyz[0] - hand_xyz[0], target_xyz[1] - hand_xyz[1], target_xyz[2] - hand_xyz[2]]
                dis_target2hand = ( target2hand[0]**2 + target2hand[1]**2 + target2hand[2]**2 ) **0.5
                with self.lock:
                    self.grasping_dis = 0.05 + 0.1*np.tanh(100*dis_target2hand)
                if dis_target2hand <= 0.03:
                    with self.lock:
                        self.grasping_dis = 0.05
                        self.status = "placing"

            elif status == "placing":
                with self.lock:
                    joints = self.motor.joints.copy()
                    joints_increment = self.motor.joints_increment.copy()
                joints = [ np.radians(joints[i]) for i in range(len(joints))]
                joints_increment[8] = (np.radians(30)*0.98 + joints[8]*0.02) - joints[8]
                with self.lock:
                    self.motor.joints_increment[8] = joints_increment[8]
                
                if joints[8] <= np.radians(35.6):
                    with self.lock:
                        self.status = "wait_to_grasp"
                        self.grasping_dis = 0.0
                        self.pos_target = self.pos_initpnt.copy()
                        self.pos_guide = self.pos_target.copy()
                elif joints[8] <= np.radians(40):
                    with self.lock:
                        self.pos_target = self.pos_initpnt.copy()
                        self.pos_guide = self.pos_target.copy()


            sys.stdout.write(f"\rstatus: {status}        ")
            sys.stdout.flush()

    def thread_camera(self):
        self.head_camera.start()
        while not self.stop_event.is_set():
            # 50 Hz
            time.sleep(0.01)

            self.head_camera.get_img(rgb=True, depth=True)
            # self.head_camera.get_target(depth=True)
            # self.head_camera.get_hand(depth=True)
            # self.head_camera.show(rgb=True, depth=False)

            with self.lock:
                status = self.status
            
            if status == "wait_to_grasp" or status == "move_to_grasp" or status == "placing":
                self.head_camera.get_target()
                self.head_camera.show(rgb=True, depth=False)
                if self.head_camera.target_exist == True:
                    with self.lock:
                        self.target_exist = True
                        self.track_done = False
                        # self.target_position_to_camera = self.head_camera.target_position.copy()
                        # self.target_pixel_norm = self.head_camera.target_norm

                        self.motor.joints_increment[0] = ( -1.5*self.head_camera.target_norm[0] - 0.4*self.motor.joints_increment[0] )
                        self.motor.joints_increment[1] = ( -1.5*self.head_camera.target_norm[1] - 0.4*self.motor.joints_increment[1] )
                    if np.abs(self.head_camera.target_norm[0]) <= 0.05 and np.abs(self.head_camera.target_norm[1]) <= 0.05 :
                        with self.lock:
                            self.target_depth = self.head_camera.target_depth
                            self.target_position_to_camera = self.head_camera.target_position.copy()
                            self.track_done = True
                else:
                    with self.lock:
                        self.target_exist = False
                        self.track_done = False
                        self.motor.joints_increment[0] = 0
                        self.motor.joints_increment[1] = 0


            elif status == "grasping" or status == "carrying" or status == "move_to_place":
                self.head_camera.get_hand(depth=True, hand="Left")
                self.head_camera.show(rgb=True, depth=False)
                if self.head_camera.hand_exist == True:
                    with self.lock:
                        self.target_exist = True
                        self.track_done = False
                        # self.target_pixel_norm = self.head_camera.hand_norm
                        self.motor.joints_increment[0] = ( -1.5*self.head_camera.hand_norm[0] - 0.4*self.motor.joints_increment[0] )
                        self.motor.joints_increment[1] = ( -1.5*self.head_camera.hand_norm[1] - 0.4*self.motor.joints_increment[1] )
                    if np.abs(self.head_camera.hand_norm[0]) <= 0.05 and np.abs(self.head_camera.hand_norm[1]) <= 0.05 :
                        with self.lock:
                            self.target_depth = self.head_camera.hand_depth
                            self.target_position_to_camera = self.head_camera.target_position.copy()
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
            time.sleep(0.02)

            with self.lock: 
                joints=self.motor.joints.copy()
                hand_length = self.grasping_dis
                track_done = self.track_done
                target_position_to_camera = self.target_position_to_camera.copy()
            joints = [ np.radians(joints[i]) for i in range(len(joints))]

    
            # Farward Kinematics of EE position
            pos_hand = self.DH_table_R.forward_hand(angles=joints[2:7].copy(), hand_length=hand_length)
            with self.lock:
                self.pos_hand = pos_hand.copy()
            if self.reachable(pos_hand.copy()) == False:
                with self.lock:
                    self.status = "wait_to_grasp"
                    self.grasping_dis = 0.0
                    self.pos_target = self.pos_initpnt.copy()
                    self.pos_guide = self.pos_target.copy()
                
            # Farward Kinematics of target position
            if track_done:
                camera_point = self.DH_table_neck.forward_neck_new(angles=joints[0:2].copy(), target_position=target_position_to_camera)
                with self.lock:
                    self.pos_grasppnt = camera_point.copy()
                    self.pos_placepnt = camera_point.copy()
            else:
                with self.lock:
                    self.pos_grasppnt = [ 0.000, -0.2488,  0.000]
                    self.pos_placepnt = [ 0.000, -0.2488,  0.000]

            # Elbow rotation
            with self.lock:
                NP_rate = self.NP_rate
                guide_xyz = self.pos_guide.copy()
                joints_increment = self.motor.joints_increment.copy()
            with torch.no_grad():
                desire_joints = self.IK(torch.tensor(guide_xyz.copy(), dtype=torch.float32)).tolist()
            NP_min = desire_joints[0]
            NP_max = min(desire_joints[0]+90, 90)
            desire_posture = np.radians( NP_min*NP_rate + NP_max*(1-NP_rate) )
            new_joint_increment = np.degrees( ( joints[5]*0.9 + desire_posture*0.1 ) - joints[5] )*0.2
            if abs(new_joint_increment) > joints_increment[5]:
                joints_increment[5] = 0.1*new_joint_increment + 0.9*joints_increment[5]
            else:
                joints_increment[5] = np.degrees( ( joints[5]*0.9 + desire_posture*0.1 ) - joints[5] ) # elbow yaw
            with self.lock:
                self.motor.joints_increment[5] = joints_increment[5]

    def thread_RL_move(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)
            with self.lock:
                joints = self.motor.joints.copy()
                joints_increment = self.motor.joints_increment.copy()
                hand_xyz = self.pos_hand.copy()
                guide_xyz = self.pos_guide.copy()
                action_old = self.RL_moving_action.copy()
                hand_length = self.grasping_dis

            # RL model1: shoulder + elbow pitch
            guidetohand = [guide_xyz[0]-hand_xyz[0], guide_xyz[1]-hand_xyz[1], guide_xyz[2]-hand_xyz[2]]
            distoguide = (guidetohand[0]**2 + guidetohand[1]**2 + guidetohand[2]**2) ** 0.5
            guidetohand_norm = guidetohand.copy()
            if distoguide >= 0.05:
                guidetohand_norm = [guidetohand[0]/distoguide*0.05, guidetohand[1]/distoguide*0.05, guidetohand[2]/distoguide*0.05]
            
            joints = [ np.radians(joints[i]) for i in range(len(joints))]
            state = np.concatenate([guide_xyz.copy(), guidetohand_norm.copy(), action_old[0:3], joints[2:4], joints[5:7], [joints[5]], [hand_length]]).astype(np.float32)
            action, _ = self.RL_moving_policy.predict(state)
            # action_new = [action_old[0]*0.9 + action[0]*0.1,
            #               action_old[1]*0.9 + action[1]*0.1,
            #               action_old[2]*0.8 + action[2]*0.2]   
            for i in range(len(action)):
                action_old[i] = action_old[i] + action[i]*0.10
                if action_old[i] > 1:       action_old[i] = 1
                elif action_old[i] < -1:    action_old[i] = -1
                 
            alpha = 1-0.5*np.exp(-100*distoguide**2)
            alpha = 1.0
            joints_increment[2] = np.degrees( action_old[0]* 0.01*alpha ) # shoulder pitch
            joints_increment[3] = np.degrees( action_old[1]* 0.01*alpha ) # shoulder roll
            joints_increment[6] = np.degrees( action_old[2]* 0.01*alpha ) # elbow pitch
        
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
        threads = [ threading.Thread(target=self.thread_status),
                    threading.Thread(target=self.thread_camera),
                    threading.Thread(target=self.thread_motor),
                    threading.Thread(target=self.thread_system),
                    threading.Thread(target=self.thread_RL_move)]
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
            hand_length = self.grasping_dis
        point_xyz = point.copy()
        max_length = 0.3708 + 0.17 + hand_length
        min_length = ( 0.3708**2 + (0.17 + hand_length)**2 -2*0.3708*(0.17 + hand_length)*np.cos(np.radians(110)) )**2
        dis2shoulder = ( (point_xyz[0]-pos_shoulder[0])**2 + (point_xyz[1]-pos_shoulder[1])**2 + (point_xyz[2]-pos_shoulder[2])**2 ) **0.5
        if dis2shoulder >= max_length or dis2shoulder <= min_length or point_xyz[0] < -0.05 or point_xyz[1] < -0.2488-(0.3708 + 0.17) or point_xyz[2] > 0.05:
            return False
        else:
            return True

if __name__ == "__main__":
    Roly = Robot_system()
    Roly.run(endtime=30)
