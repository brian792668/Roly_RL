import sys
import os
import threading
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import SAC  
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info
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
        self.color_img = self.head_camera.color_img
        self.depth_colormap = self.head_camera.depth_colormap

        # Initial RL
        RL_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RLmodel/model_1/v21-30/model.zip")
        self.RL_model1  = SAC.load(RL_path1)
        self.RL_state   = [0] * 7
        self.RL_action  = [0] * 6

        self.IK = IKMLP()
        self.IK.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "IKmodel_v7.pth"), weights_only=True))
        self.IK.eval()

        # Initial motors
        self.joints = [0] * 8
        self.motor = init_motor()
        self.joints = initial_pos(self.motor)
        self.limit_high = [ 1.57, 0.12, 1.57, 1.90]
        self.limit_low  = [-1.05,-1.57,-1.57, 0.00]

        # Initial mechanism
        self.target_pos   = [0.19, -0.22, -0.36]
        self.hand_pos     = [0.19, -0.22, -0.36]
        self.shoulder_pos = [-0.02, -0.2488, -0.104]
        self.elbow_pos    = [-0.02, -0.2488, -0.35]
        self.arm_target_joints = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.DH_table_R = DHtable([[    0.0, np.pi/2,   -0.02,  -0.104],
                                   [np.pi/2, np.pi/2,     0.0,  0.2488],
                                   [    0.0,     0.0, -0.1105,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.1403],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1803]])
        self.DH_table_L = DHtable([[    0.0, np.pi/2,   -0.02,  -0.104],
                                   [np.pi/2, np.pi/2,     0.0, -0.2488],
                                   [    0.0,     0.0, -0.1105,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.1403],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1803]])
        self.DH_table_elbow = DHtable([[    0.0, np.pi/2,   -0.02,  -0.104],
                                       [np.pi/2, np.pi/2,     0.0,  0.2488],
                                       [    0.0,     0.0, -0.1105,     0.0],
                                       [np.pi/2, np.pi/2,     0.0,     0.0],
                                       [np.pi/2, np.pi/2,     0.0, -0.1403]])
        
        self.blue_line_points = [0, 1, 2, 3, 4]  # 藍線連接 point1 ~ point5
        self.red_line_points = [4, 5]  # 紅線連接 point5 ~ point6

    def camera_thread(self):
        self.head_camera.start()
        while not self.stop_event.is_set():
            # 50 Hz
            time.sleep(0.01)

            self.head_camera.get_img(rgb=True, depth=True)
            self.head_camera.get_target(depth=True)
            # self.head_camera.show(rgb=True, depth=True)
            with self.lock:
                self.color_img = self.head_camera.color_img
                self.depth_colormap = self.head_camera.depth_colormap
            cv2.imshow("Realsense D435i RGB", self.color_img)
            cv2.imshow("Realsense D435i Depth with color", self.depth_colormap)
            cv2.waitKey(1)

            if self.head_camera.target_exist == True:
                with self.lock:
                    self.target_exist = True
                    self.target_pixel_norm = self.head_camera.target_norm
                if np.abs(self.head_camera.target_norm[0]) <= 0.1 and np.abs(self.head_camera.target_norm[1]) <= 0.1 :
                    with self.lock:
                        self.target_depth = self.head_camera.target_depth
            else:
                with self.lock:
                    self.target_exist = False
        
        self.head_camera.stop()

    def system_thread(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.02)

            # Farward Kinematics of EE position
            with self.lock: angles=self.joints.copy()
            angles = [ np.radians(angles[i]) for i in range(len(angles))]
            elbow_pos = self.DH_table_elbow.forward2(angles=angles[2:6].copy())
            hand_pos = self.DH_table_R.forward(angles=angles[2:7].copy())
            # print(f"{hand_pos[0]:.2f}, {hand_pos[1]:.2f}, {hand_pos[2]:.2f}")

            # Farward Kinematics of target position
            with self.lock:
                target_pos = self.target_pos.copy()
                self.hand_pos = hand_pos.copy()
                shoulder_pos = self.shoulder_pos.copy()
                
            neck_angle = angles[0]
            camera_angle = angles[1]
            with self.lock: 
                target_exist = self.target_exist
                d = self.target_depth + 0.01
            if target_exist:
                a = 0.06
                b = (a**2 + d**2)**0.5
                beta = np.arctan2(d, a)
                gamma = np.pi/2 + camera_angle-beta
                d2 = b*np.cos(gamma)
                z = b*np.sin(gamma)
                x = d2*np.cos(neck_angle)
                y = d2*np.sin(neck_angle)
                if self.reachable([x,y,z].copy()) == True: 
                    target_pos = [x, y, z] 
                # print(f"{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}")

            # Farward Kinematics of target elbow
            elbow_pos = self.DH_table_elbow.forward2(angles=angles[2:6].copy())
            # print(f"{elbow_pos[0]:.2f}, {elbow_pos[1]:.2f}, {elbow_pos[2]:.2f}")
            with self.lock:
                self.elbow_pos = elbow_pos.copy()
                self.target_pos = target_pos.copy()

    def RL_thread(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)
            with self.lock:
                joints = self.joints.copy()
                object_xyz = self.target_pos.copy()
                hand_xyz = self.hand_pos.copy()
                action_old = self.RL_action.copy()
                timenow = time.time() - self.time_start

            target_to_EE = [object_xyz[0]-hand_xyz[0], object_xyz[1]-hand_xyz[1], object_xyz[2]-hand_xyz[2]]
            distotarget = (target_to_EE[0]**2 + target_to_EE[1]**2 + target_to_EE[2]**2) ** 0.5
            target_to_EE_norm = [target_to_EE[0]/distotarget*0.02, target_to_EE[1]/distotarget*0.02, target_to_EE[2]/distotarget*0.02]
            # print(target_to_EE_norm)
            # alpha = 1-0.8*np.exp(-100*distotarget**2)
            
            joints = [ np.radians(joints[i]) for i in range(len(joints))]
            state = np.concatenate([object_xyz.copy(), target_to_EE_norm.copy(), action_old[0:2], action_old[4:5], joints[2:4], joints[5:7]]).astype(np.float32)
            # print(f"{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}")

            with torch.no_grad():  # 不需要梯度計算，因為只做推論
                desire_joints = self.IK(torch.tensor(object_xyz.copy(), dtype=torch.float32)).tolist()
                desire_joints = np.radians(desire_joints)
            # desire_joints[2] = np.radians(45+45*np.sin(timenow*2))
            desire_joints[2] = np.radians(20)

            action, _ = self.RL_model1.predict(state)
            # print(action)
            action_new = [action_old[0]*0.9 + action[0]*0.1,
                          action_old[1]*0.9 + action[1]*0.1,  
                          action_old[2]*0.9 + 0, 
                          action_old[3]*0.9 + 0,
                          action_old[4]*0.9 + action[2]*0.1,
                          action_old[5]*0.9 + 0]          

            # joints[2] += action_new[0]*0.08*alpha
            # joints[3] += action_new[1]*0.05
            # joints[3] = joints[3]*0.95+ desire_joints[1]*0.05
            # joints[4] += action_new[2]*0.08*alpha
            # joints[5] += action_new[3]*0.08*alpha
            # joints[6] += action_new[4]*0.08*alpha
            # joints[7] += action_new[5]*0.05*alpha

            # joints[2] = joints[2]*0.95 + desire_joints[0]*0.05
            # joints[3] = joints[3]*0.95 + desire_joints[1]*0.05
            joints[5] = joints[5]*0.9 + desire_joints[2]*0.1
            # joints[6] = joints[6]*0.95 + desire_joints[3]*0.05

            joints[2] += action_new[0]* 0.03
            joints[3] += action_new[1]* 0.03
            joints[4] += action_new[2]* 0.03
            joints[5] += action_new[3]* 0.03
            joints[6] += action_new[4]* 0.03
            joints[7] += action_new[5]* 0.03

        
            if   joints[2] > self.limit_high[0]: joints[2] = self.limit_high[0]
            elif joints[2] < self.limit_low[0] : joints[2] = self.limit_low[0]
            if   joints[3] > self.limit_high[1]: joints[3] = self.limit_high[1]
            elif joints[3] < self.limit_low[1] : joints[3] = self.limit_low[1]
            if   joints[5] > self.limit_high[2]: joints[5] = self.limit_high[2]
            elif joints[5] < self.limit_low[2] : joints[5] = self.limit_low[2]
            if   joints[6] > self.limit_high[3]: joints[6] = self.limit_high[3]
            elif joints[6] < self.limit_low[3] : joints[6] = self.limit_low[3]

            joints = [ np.degrees(joints[i]) for i in range(len(joints))]
            print(joints[2:7])
            
            with self.lock:
                self.RL_action = action_new.copy()
                self.joints = joints.copy()

    def motor_thread(self):
        # with self.lock:
        #     self.joints = self.motor.readAllMotorPosition()
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)
            with self.lock: 
                joints = self.joints.copy()
                pixel_norm = self.target_pixel_norm.copy()
                target_exist = self.target_exist

            joints[0] += -3.0*pixel_norm[0]*target_exist
            joints[1] += -3.0*pixel_norm[1]*target_exist
            with self.lock: self.joints = joints.copy()
            self.motor.writeAllMotorPosition(self.motor.toRolyctrl(joints.copy()))
        
        self.motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[100]*8)
        with self.lock: joints = self.joints.copy()
        t=0
        while t <= 1.0:
            joints, t = smooth_transition(t, initial_angles=joints.copy(), final_angles=[0]*8, speed=0.002)
            self.motor.writeAllMotorPosition(self.motor.toRolyctrl(joints.copy()))
            time.sleep(0.01)

        self.motor.setAllMotorTorqurDisable()
        self.motor.portHandler.closePort()

    def run(self, endtime = 10):     
        threads = [
                threading.Thread(target=self.motor_thread),
                threading.Thread(target=self.system_thread),
                threading.Thread(target=self.camera_thread),
                threading.Thread(target=self.RL_thread)
            ]

        for t in threads:
            t.start()

        with self.lock: 
            time0 = self.time_start
        while not self.stop_event.is_set():
            if time.time() - time0 >= endtime:  # 執行 10 秒後結束
                self.stop_event.set()
            time.sleep(0.02)  # 減少CPU負擔

        for t in threads:
            t.join()

        cv2.destroyAllWindows()
        # plt.close()

        print("Program Done.\n")

    def plot(self):
        # plt.clf()
        # plt.close('all')

        with self.lock:
            points_x = self.points_x.copy()
            points_y = self.points_y.copy()
            points_z = self.points_z.copy()
        # 創建三維繪圖
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 繪製藍線連接點1~點5
        ax.plot(
            [points_x[j] for j in self.blue_line_points],
            [points_y[j] for j in self.blue_line_points],
            [points_z[j] for j in self.blue_line_points],
            color='blue',
            label='Line Point1-Point5'
        )

        # 繪製紅線連接點5~點6
        ax.plot(
            [points_x[j] for j in self.red_line_points],
            [points_y[j] for j in self.red_line_points],
            [points_z[j] for j in self.red_line_points],
            color='red',
            label='Line Point5-Point6'
        )

        # 繪製所有點
        ax.scatter(points_x, points_y, points_z, color='black', label='Points')

        # 設置視角
        ax.view_init(elev=30, azim=-30)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Default View (30, 30)')

        # # 儲存圖形為檔案
        save_path = "3d_plot.png"
        plt.savefig(save_path)  # 設定解析度 dpi
        plt.close()  # 關閉 Matplotlib 圖形，節省記憶體
        # time.sleep(0.01)

        plt.clf()
        plt.close('all')

    def reachable(self, target):
        with self.lock:
            shoulder_pos = self.shoulder_pos.copy()
        target_pos = target.copy()
        distoshoulder = ( (target_pos[0]-shoulder_pos[0])**2 + (target_pos[1]-shoulder_pos[1])**2 + (target_pos[2]-shoulder_pos[2])**2 ) **0.5
        # distoshoulder = ( (point[0]-0.00)**2 + (point[1]+0.25)**2 + (point[2]-1.35)**2 ) **0.5
        if distoshoulder >= 0.45 or distoshoulder <= 0.25:
            return False
        else:
            return True

    # def smooth_transition(self, t, initial_angles, final_angles, speed=0.001):

        initial_angles = np.array(initial_angles)
        final_angles = np.array(final_angles)

        progress = min(t, 1)
        progress = ((1 - np.cos(np.pi * progress)) / 2)**2
        current_angles = initial_angles*(1-progress) + final_angles*progress

        t_next = t + speed
        return current_angles.tolist(), t_next

if __name__ == "__main__":

    Roly = Robot_system()
    Roly.run(endtime=30)
