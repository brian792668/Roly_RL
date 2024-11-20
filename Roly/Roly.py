import sys
import os
import threading
import numpy as np
import time
from stable_baselines3 import SAC  # 已訓練的RL模型

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info
from imports.Camera import *
from imports.Forward_kinematics import *

class Robot_system:
    def __init__(self):
        # 初始化輸入參數
        self.head_camera = Camera()
        self.color_img = self.head_camera.color_img
        self.depth_img = self.head_camera.depth_img
        self.target_exist = self.head_camera.target_exist
        self.target_pixel = self.head_camera.target_pixel
        self.target_depth = self.head_camera.target_depth

        # self.motor = self.init_motor()
        self.EE_goal_pos      = [0.00, -0.25, -0.40]
        self.EE_current_pos   = [0.00, -0.25, -0.40]
        self.motor_ctrl_pos   = [0] * 8 
        self.RL_model         = SAC.load("/home/brianll/Desktop/Roly/Roly/Roly/Sim2Real/model1/v3_current_model.zip")
        self.RL_state         = [0] * 7
        self.RL_action        = [0] * 8
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
        time.sleep(0.1)
        motor.pos_ctrl = motor.toRolyctrl([0, 0, 0, 0, 0, 0, 0, 0])
        motor.writeAllMotorPosition(motor.pos_ctrl)
        time.sleep(3)
        
        return motor

    def camera_thread(self):
        while not self.stop_event.is_set():
            self.head_camera.get_img(rgb=True, depth=True)
            self.head_camera.get_depth()
            self.head_camera.show(rgb=True, depth=True)

            with self.lock:
                self.color_img = self.head_camera.color_img
                self.depth_img = self.head_camera.depth_img

            if self.head_camera.target_exist == True:
                with self.lock:
                    self.target_exist = True
                    self.target_depth = self.head_camera.target_depth
                    self.target_pixel = self.head_camera.target_pixel
            else:
                with self.lock:
                    self.target_exist = False
                
    def compute_thread(self):
        while not self.stop_event.is_set():
            # with self.lock:
                # 模擬正向運動學計算更新手部位置
            EE_goal_pos = [0.20, -0.25, -0.20]  # 模擬計算
            EE_current_pos = self.DH_table_R.forward(angles=self.motor_ctrl_pos[2:8])
            print(EE_current_pos)
            time.sleep(1)
            with self.lock:
                self.EE_goal_pos = EE_goal_pos  # 模擬計算
                self.EE_current_pos = EE_current_pos
            print(self.EE_current_pos)
            time.sleep(1)

    def rl_thread(self):
        while not self.stop_event.is_set():
            with self.lock:
                # 模擬使用RL模型產生動作
                observation = self.motor_angles[2:7] + self.hand_goal_pos
                self.rl_action = self.rl_model.predict(observation)[0]  # 假設模型輸出

    def motor_command_thread(self):
        while not self.stop_event.is_set():
            with self.lock:
                # 更新馬達控制角度
                self.motor_angles = [o + a for o, a in zip(self.motor_angles, self.rl_action)]

    def run(self):     
        threads = [
            threading.Thread(target=self.camera_thread),
            threading.Thread(target=self.compute_thread),
            # threading.Thread(target=self.rl_thread),
            # threading.Thread(target=self.motor_command_thread)
        ]

        start_time = time.time()
        for t in threads:
            t.start()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > 10:  # 執行 5 秒後結束
                self.stop_event.set()  # 通知所有執行緒停止
                break
            time.sleep(0.1)  # 減少CPU負擔

        for t in threads:
            t.join()

        cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
        print("Program Done.")

# 建立Roly實例並啟動
Roly = Robot_system()
Roly.run()
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
