import sys
import os
import threading
import numpy as np
import time
from stable_baselines3 import SAC  

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info
from imports.Camera import *
from imports.Forward_kinematics import *

class Robot_system:
    def __init__(self):
        # Initial camera
        self.head_camera = Camera()
        self.target_exist = self.head_camera.target_exist
        self.target_depth = self.head_camera.target_depth
        self.target_pixel_norm = self.head_camera.target_norm

        # Initial RL
        RL_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RL_model.zip")
        self.RL_model   = SAC.load(RL_path)
        self.RL_state   = [0] * 7
        self.RL_action  = [0] * 8

        # Initial motors
        self.joints     = [0] * 8
        self.motor = self.init_motor()

        # Initial mechanism
        self.EE_goal_pos      = [0.00, -0.25, -0.40]
        self.EE_current_pos   = [0.00, -0.25, -0.40]
        self.DH_table_R = DHtable([[    0.0, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,  0.2488],
                                   [    0.0,     0.0, -0.1105,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.1195],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1803]])
        self.DH_table_L = DHtable([[    0.0, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.2488],
                                   [    0.0,     0.0, -0.1105,     0.0],
                                   [np.pi/2, np.pi/2,     0.0,     0.0],
                                   [np.pi/2, np.pi/2,     0.0, -0.1195],
                                   [    0.0, np.pi/2,     0.0,     0.0],
                                   [    0.0, np.pi/2,     0.0,  0.1803]])
        
        # Initial system
        self.system_running = True
        self.time_start = time.time()
        
        # Lock物件用於多執行緒資源管理
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
    def init_motor(self):
        X_series_info = X_Motor_Info()
        P_series_info = P_Motor_Info()

        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {
            "id": [1, 2, 10, 11, 12, 13, 14, 15], 
            "model": [X_series_info] * 8
        }

        motor = DXL_Motor(DEVICENAME, DXL_MODELS, BAUDRATE=115200)
        motor.changeAllMotorOperatingMode(OP_MODE=3)
        motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[30]*len(motor.pos_ctrl))
        motor.setAllMotorTorqueEnable()
        motor.writeAllMotorPosition(motor.toRolyctrl(self.joints))
        time.sleep(0.1)
        
        return motor

    def camera_thread(self):
        while not self.stop_event.is_set():
            # 50 Hz
            time.sleep(0.02)

            self.head_camera.get_img(rgb=True, depth=True)
            self.head_camera.get_target(depth=True)
            self.head_camera.show(rgb=True, depth=True)

            if self.head_camera.target_exist == True:
                with self.lock:
                    self.target_exist = True
                    self.target_depth = self.head_camera.target_depth
                    self.target_pixel_norm = self.head_camera.target_norm
                    # print(f"{self.target_pixel_norm[0]:.2f}, {self.target_pixel_norm[1]:.2f}")
            else:
                with self.lock:
                    self.target_exist = False
                
    def system_thread(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)

            # clock
            # with self.lock: clock = time.time() - self.time_start
            # print(f"{clock:.1f}")

            # Farward Kinematics of EE position
            with self.lock: angles=self.joints
            angles = [ np.radians(angles[i]) for i in range(len(angles))]
            EE_current_pos = self.DH_table_R.forward(angles=angles[2:8])
            # print(f"{EE_current_pos[0]:.2f}, {EE_current_pos[1]:.2f}, {EE_current_pos[2]:.2f}")

            # Farward Kinematics of target position
            EE_goal_pos = [0.20, -0.25, -0.30]
            neck_angle = angles[0]
            camera_angle = angles[1]
            with self.lock: 
                target_exist = self.target_exist
                d = self.target_depth
            if target_exist and 0.70 >= d >= 0.15:
                a = 0.06
                b = (a**2 + d**2)**0.5
                beta = np.arctan2(d, a)
                gamma = np.pi/2 + camera_angle-beta
                d2 = b*np.cos(gamma)
                z = b*np.sin(gamma)
                x = d2*np.cos(neck_angle)
                y = d2*np.sin(neck_angle)
                EE_goal_pos = [x, y, z]
                # print(f"{EE_goal_pos[0]:.2f}, {EE_goal_pos[1]:.2f}, {EE_goal_pos[2]:.2f}")

            with self.lock:
                self.EE_goal_pos = EE_goal_pos
                self.EE_current_pos = EE_current_pos

    def RL_thread(self):
        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)
            with self.lock:
                observation = self.motor_angles[2:7] + self.hand_goal_pos
                self.rl_action = self.RL_model.predict(observation)[0]

    def motor_thread(self):
        # initial
        with self.lock: angles = self.joints
        angles = [0, 0, -30, 0, 0, 0, 60, 0]

        angles[1] *= -1
        angles[5] *= -1
        angles[6] *= -1
        self.motor.writeAllMotorPosition(self.motor.toRolyctrl(angles))
        angles[1] *= -1
        angles[5] *= -1
        angles[6] *= -1
        with self.lock: self.joints = angles
        time.sleep(3)
        self.motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[100, 100, 30, 30, 30, 30, 30, 30])

        while not self.stop_event.is_set():
            # 100 Hz
            time.sleep(0.01)
            with self.lock: angles = self.joints
            with self.lock: 
                pixel_norm = self.target_pixel_norm
                target_exist = self.target_exist

            angles[0] += -1.5*pixel_norm[0]*target_exist
            angles[1] += -1.5*pixel_norm[1]*target_exist
            # angles[2] += 0
            # angles[3] += 0
            # angles[4] += 0
            # angles[5] += 0
            # angles[6] += 0

            angles[1] *= -1
            angles[5] *= -1
            angles[6] *= -1
            self.motor.writeAllMotorPosition(self.motor.toRolyctrl(angles))
            angles[1] *= -1
            angles[5] *= -1
            angles[6] *= -1

            with self.lock: self.joints = angles

    def run(self):     
        threads = [
            threading.Thread(target=self.motor_thread),
            threading.Thread(target=self.system_thread),
            threading.Thread(target=self.camera_thread),
            # threading.Thread(target=self.RL_thread),
        ]

        for t in threads:
            t.start()

        while not self.stop_event.is_set():
            self.time_now = time.time() - self.time_start
            if self.time_now > 10:  # 執行 10 秒後結束
                self.stop_event.set()
            time.sleep(0.1)  # 減少CPU負擔

        for t in threads:
            t.join()

        cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
        self.head_camera.stop()
        self.motor.setAllMotorTorqurDisable()
        self.motor.portHandler.closePort()
        print("Program Done.\n")

Roly = Robot_system()
Roly.run()
