from PyQt5 import QtWidgets, QtCore
import UI
import cv2
from stable_baselines3 import SAC  
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info
from imports.Camera import *
from imports.Roly_motor import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = UI.Roly_UI(self)
        self.setup_button()

        # System
        self.system_timer = QtCore.QTimer(self)
        self.system_timer.timeout.connect(self.System_update)
        self.system_timer.start(100)

        # Initial Camera
        self.camera = Camera()
        self.camera_timer = QtCore.QTimer(self)
        self.camera_timer.timeout.connect(self.Camera_update)

        self.target_exist = False
        self.target_pixel_norm = self.camera.target_norm
        self.target_depth = self.camera.target_depth

        # Initial RL
        self.RL_is_running = False
        self.RL_timer = QtCore.QTimer(self)
        self.RL_timer.timeout.connect(self.RL_update)
        RL_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RLmodel/model_1/v17/model.zip")
        self.RL_model1  = SAC.load(RL_path1)
        self.RL_action  = [0] * 6


        # Initial motor
        self.joints = [0] * 8
        self.motor = init_motor()
        self.motor_is_running = False
        self.torque_is_on = False
        self.motor_timer = QtCore.QTimer(self)
        self.motor_timer.timeout.connect(self.Motor_update)
        self.xyz_target = [0, 0, 0]
        self.xyz_shoulder = [0, 0, 0]
        self.xyz_hand = [0, 0, 0]

    def setup_button(self):
        self.ui.camera_start.clicked.connect(self.Camera_start)
        self.ui.camera_stop.clicked.connect(self.Camera_stop)

        self.ui.RL_start.clicked.connect(self.RL_start)
        self.ui.RL_stop.clicked.connect(self.RL_stop)

        self.ui.motor_start.clicked.connect(self.Motor_start)
        self.ui.motor_stop.clicked.connect(self.Motor_stop)
        self.ui.motor_init.clicked.connect(self.Motor_init)
        self.ui.motor_on.clicked.connect(self.Motor_on)
        self.ui.motor_off.clicked.connect(self.Motor_off)

        self.ui.stop_all.clicked.connect(self.Stop_all)

    def System_update(self):
        # camera status
        if self.camera.is_running == True: self.ui.camera_status_light.setStyleSheet("""background-color: red;  border-radius: 5px;  border: 1px solid white; """)
        else: self.ui.camera_status_light.setStyleSheet("""background-color: gray;  border-radius: 5px;  border: 2px solid white; """)

        # RL status
        if self.RL_is_running == True: self.ui.RL_status_light.setStyleSheet("""background-color: red;  border-radius: 5px;  border: 1px solid white; """)
        else: self.ui.RL_status_light.setStyleSheet("""background-color: gray;  border-radius: 5px;  border: 2px solid white; """)

        # motor status
        if self.motor_is_running == True: self.ui.motor_status_light.setStyleSheet("""background-color: red;  border-radius: 5px;  border: 1px solid white; """)
        else: self.ui.motor_status_light.setStyleSheet("""background-color: gray;  border-radius: 5px;  border: 2px solid white; """)

        # torque status
        if self.torque_is_on == True: self.ui.torque_status_light.setStyleSheet("""background-color: red;  border-radius: 5px;  border: 1px solid white; """)
        else: self.ui.torque_status_light.setStyleSheet("""background-color: gray;  border-radius: 5px;  border: 2px solid white; """)


    # Camera
    def Camera_start(self):
        if self.camera.is_running != True:
            self.camera_timer.start(20)
            self.camera.start()
            self.camera.is_running = True

    def Camera_stop(self):
        if self.camera.is_running == True:
            self.camera_timer.stop()
            self.camera.stop()
            self.camera.is_running = False

    def Camera_update(self):
        self.camera.get_img(rgb=True, depth=True)
        self.camera.get_target(depth=True)
        self.camera.show(rgb=True, depth=False)

        if self.camera.target_exist == True:
            self.target_exist = True
            self.target_pixel_norm = self.camera.target_norm
            if np.abs(self.camera.target_norm[0]) <= 0.1 and np.abs(self.camera.target_norm[1]) <= 0.1 :
                self.target_depth = self.camera.target_depth
        else:
            self.target_exist = False

    # RL
    def RL_start(self):
        if self.RL_is_running != True:
            self.RL_timer.start(10)
            self.RL_is_running = True

    def RL_stop(self):
        if self.RL_is_running == True:
            self.RL_timer.stop()
            self.RL_is_running = False

    def RL_update(self):
        xyz_target_to_hand = np.array(self.xyz_target.copy()) - np.array(self.xyz_hand.copy())
        state = np.concatenate([self.xyz_target.copy(), xyz_target_to_hand.copy(), self.joints[2:4], self.joints[5:7]]).astype(np.float32)
        action, _ = self.RL_model1.predict(state)
        action_new = [self.RL_action[0]*0.98 + action[0]*0.02,
                      self.RL_action[1]*0.98 + action[1]*0.02,  
                      self.RL_action[2]*0.95 + 0, 
                      self.RL_action[3]*0.95 + 0,
                      self.RL_action[4]*0.95 + action[2]*0.05,
                      self.RL_action[5]*0.95 + 0]
        
        self.joints[2] += action_new[0]**3 * 0.05
        self.joints[3] += action_new[1]**3 * 0.05
        self.joints[4] += action_new[2]**3 * 0.05
        self.joints[5] += action_new[3]**3 * 0.05
        self.joints[6] += action_new[4]**3 * 0.05
        self.joints[7] += action_new[5]**3 * 0.05

        self.RL_action = action_new.copy()
        print("RL running")
        # print(self.joints)

    # Motor
    def Motor_start(self):
        self.Motor_on()
        if self.motor_is_running != True:
            self.motor_timer.start(10)
            self.motor_is_running = True

    def Motor_stop(self):
        if self.motor_is_running == True:
            self.motor_timer.stop()
            self.motor_is_running = False
            # self.motor.portHandler.closePort()

    def Motor_init(self):
        self.motor_is_running = False
        self.Motor_on()
        self.joints = initial_pos(self.motor)

    def Motor_on(self):
        self.motor.changeAllMotorOperatingMode(OP_MODE=3)
        self.torque_is_on = True

    def Motor_off(self):
        self.motor.setAllMotorTorqurDisable()
        self.torque_is_on = False
        self.motor_is_running = False

    def Motor_update(self):
        print("motor running")
        pass

    def Stop_all(self):
        if self.camera.is_running == True:
            self.camera_timer.stop()
            self.camera.stop()
            self.camera.is_running = False
        
        if self.RL_is_running == True:
            self.RL_timer.stop()
            self.RL_is_running = False

        if self.motor_is_running == True:
            self.motor_timer.stop()
            self.motor_is_running = False