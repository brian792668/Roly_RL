import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info

class Roly_motor(DXL_Motor):
    # def __init__(self):
    #     X_series_info = X_Motor_Info()
    #     DEVICENAME = "/dev/ttyUSB0"
    #     DXL_MODELS = {  "id": [1, 2,
    #                            10, 11, 12, 13, 14, 15, 16,
    #                            20, 21, 22, 23, 24, 25, 26], 
    #                     "model": [X_series_info] * 16 }
    #     super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

    #     self.joints_bias = [184, 180,
    #                         180, 180, 180, 180, 180, 180, 180,
    #                         180, 180, 180, 180, 180, 180, 180]
    #     self.joints_axis = [ 1, -1,
    #                          1,  1,  1, -1, -1,  1,  1,
    #                         -1, -1, -1,  1, -1, -1,  1]
    #     self.joints = [ 0, 0,
    #                     0, 0, 0, 0, 0, 0, 90,
    #                     0, 0, 0, 0, 0, 0, 90]
    #     self.joints_increment = [0] * 16
    #     # self.initial_pos = [-20, -45, -11, -26, 0, 11, 90, 90, 90] # gripper closed
    #     self.initial_pos = [   0,   0,
    #                          -11, -26,  0, 11, 90, 90, 90,  # gripper opened: 90 degree
    #                            0,   0,  0,  0, 90,  0, 90]  # gripper closed: 5 degree
    #     self.limit_high = [ 1.57, 0.00, 1.57, 1.90]
    #     self.limit_low  = [-1.57,-1.57,-1.57, 0.00]

    #     self.changeAllMotorOperatingMode(OP_MODE=3)
    #     self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 100, 100,
    #                                                          40, 40, 40, 40, 40, 40, 200,
    #                                                          40, 40, 40, 40, 40, 40, 200])
    #     time.sleep(0.1)

    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [1, 2,
                               10, 11, 12, 13, 14, 15, 16], 
                        "model": [X_series_info] * 9 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [184, 180,
                            190, 180, 180, 180, 180, 180, 180]
        self.joints_axis = [ 1, -1,
                             1,  1,  1, -1, -1,  1,  1]
        self.joints = [ 0, 0,
                        0, 0, 0, 0, 0, 0, 90]
        self.joints_increment = [0] * 9
        # self.initial_pos = [-20, -45, -11, -26, 0, 11, 90, 90, 90] # gripper closed
        self.initial_pos = [   0,   0,
                             -11, -26,  0, 11, 90, 90, 5]  # gripper closed: 5 degree
        self.limit_high = [ 1.57, 0.00, 1.57, 1.90]
        self.limit_low  = [-1.57,-1.57,-1.57, 0.00]

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 100, 100,
                                                             40, 40, 40, 40, 40, 40, 200])
        time.sleep(0.1)

    def to_pose(self, pose, speed=0.5):
        # current_angles = self.joints.copy()
        # Get current joint angles.
        self.writeOperatingMode(OP_MODE=3)
        current_angles = None
        while current_angles ==  None:
            time.sleep(0.1)
            print("failed to read motor position. Retry...")
            current_angles = self.readAllMotorPosition()
        current_angles = [(resolution2degree(current_angles[i])-self.joints_bias[i])*self.joints_axis[i] for i in range(len(self.joints))]

        # Set final joint angles.
        final_angles = current_angles.copy()
        if pose == "initial":   
            final_angles = self.initial_pos.copy()
            print("\n\033[1;33m[ Motor  ]\033[0m To INITIAL pose ...")
        elif pose == "shut down": 
            # final_angles = [0]*16
            final_angles = [0]*9
            final_angles[8] = 90
            # final_angles[15] = 90
            print("\n\033[1;33m[ Motor  ]\033[0m To SHUT DOWN pose ...")

        # Generate smoot motion from cosine function.
        t=0
        np_current_angles = np.array(current_angles)
        np_final_angles = np.array(final_angles)
        while t <= 1.0:
            progress = ((1 - np.cos(np.pi * t)) / 2)
            ctrlpos = (np_current_angles*(1-progress) + np_final_angles*progress).tolist()
            t += 0.01*speed
            self.writeAllMotorPosition(self.toRolyctrl(ctrlpos.copy()))
            time.sleep(0.001)

        self.joints = final_angles.copy()

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]
    
class Roly_head(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [1, 2], 
                        "model": [X_series_info] * 2 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [184, 180]
        self.joints_axis = [ 1, -1]
        self.joints = [0, 0]
        self.joints_increment = [0, 0]
        self.initial_pos = [0, 0]

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 100, 100])
        time.sleep(0.01)

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]

    def to_pose(self, pose):
        # Set final joint angles.
        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 20, 20])
        time.sleep(0.01)
        if pose == "initial":   
            self.joints = self.initial_pos.copy()
            print("\n\033[1;33m[ Motor  ]\033[0m To INITIAL pose ...")
        elif pose == "shut down": 
            self.joints = [0, 0]
            print("\n\033[1;33m[ Motor  ]\033[0m To SHUT DOWN pose ...")

        for i in range(10):
            self.writeAllMotorPosition(self.toRolyctrl(self.joints.copy()))
            time.sleep(0.01)
    
class Roly_R_arm(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [10, 11, 13, 14, 16],
                        "model": [X_series_info] * 5 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [84.6, 180, 185, 175, 180]
        self.joints_axis = [ 1,  1, -1, -1,  1]
        self.joints = [0, 0, 0, 0, 5]
        self.joints_increment = [0, 0, 0, 0, 0]
        self.initial_pos = [ -11, -26, 11, 90, 5] # gripper opened: 5 degree
        # self.initial_pos = [ -11, -26, 11, 90, 85]  # gripper closed: 85 degree

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[60, 60, 20, 60, 200])
        time.sleep(0.01)

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]

    def to_pose(self, pose):
        # Set final joint angles.
        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 20, 20, 20, 20, 50])
        time.sleep(0.01)
        if pose == "initial":   
            self.joints = self.initial_pos.copy()
            print("\n\033[1;33m[ Motor  ]\033[0m To INITIAL pose ...")
        elif pose == "shut down": 
            self.joints = [0, 0, 0, 0, 85]
            print("\n\033[1;33m[ Motor  ]\033[0m To SHUT DOWN pose ...")

        for i in range(10):
            self.writeAllMotorPosition(self.toRolyctrl(self.joints.copy()))
            time.sleep(0.01)
    
class Roly_R_arm_others(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [12, 15], 
                        "model": [X_series_info] * 2 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [180, 180]
        self.joints_axis = [ 1, 1]
        self.joints = [0, 0]
        self.joints_increment = [0, 0]
        self.initial_pos = [0, 90]

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 40, 40])
        time.sleep(0.1)

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]

    def to_pose(self, pose):
        # Set final joint angles.
        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 20, 20])
        time.sleep(0.01)
        if pose == "initial":   
            self.joints = self.initial_pos.copy()
            print("\n\033[1;33m[ Motor  ]\033[0m To INITIAL pose ...")
        elif pose == "shut down": 
            self.joints = [0, 0]
            print("\n\033[1;33m[ Motor  ]\033[0m To SHUT DOWN pose ...")

        for i in range(10):
            self.writeAllMotorPosition(self.toRolyctrl(self.joints.copy()))
            time.sleep(0.01)

class Roly_L_arm(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [20, 21, 23, 24, 26],
                        "model": [X_series_info] * 5 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [180, 180, 180, 180, 180]
        self.joints_axis = [ -1, -1,  1, -1,  1]
        self.joints = [0, 0, 0, 0, 90]
        self.joints_increment = [0, 0, 0, 0, 0]
        self.initial_pos = [ -11, -26, 11, 90, 85] # gripper opened: 85 degree
        # self.initial_pos = [ -11, -26, 11, 90, 5]  # gripper closed: 5 degree

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[40, 40, 40, 40, 200])
        time.sleep(0.01)

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]

    def to_pose(self, pose):
        # Set final joint angles.
        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 20, 20, 20, 20, 50])
        time.sleep(0.01)
        if pose == "initial":   
            self.joints = self.initial_pos.copy()
            print("\n\033[1;33m[ Motor  ]\033[0m To INITIAL pose ...")
        elif pose == "shut down": 
            self.joints = [0, 0, 0, 0, 85]
            print("\n\033[1;33m[ Motor  ]\033[0m To SHUT DOWN pose ...")

        for i in range(10):
            self.writeAllMotorPosition(self.toRolyctrl(self.joints.copy()))
            time.sleep(0.01)
    
class Roly_L_arm_others(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [22, 25], 
                        "model": [X_series_info] * 2 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [180, 180]
        self.joints_axis = [-1, -1]
        self.joints = [0, 0]
        self.joints_increment = [0, 0]
        self.initial_pos = [0, 90]

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 40, 40])
        time.sleep(0.1)

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]

    def to_pose(self, pose):
        # Set final joint angles.
        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[ 20, 20])
        time.sleep(0.01)
        if pose == "initial":   
            self.joints = self.initial_pos.copy()
            print("\n\033[1;33m[ Motor  ]\033[0m To INITIAL pose ...")
        elif pose == "shut down": 
            self.joints = [0, 0]
            print("\n\033[1;33m[ Motor  ]\033[0m To SHUT DOWN pose ...")

        for i in range(10):
            self.writeAllMotorPosition(self.toRolyctrl(self.joints.copy()))
            time.sleep(0.01)


