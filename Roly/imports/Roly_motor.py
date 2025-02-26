import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info

class Roly_motor(DXL_Motor):
    def __init__(self):
        X_series_info = X_Motor_Info()
        DEVICENAME = "/dev/ttyUSB0"
        DXL_MODELS = {  "id": [1, 2, 10, 11, 12, 13, 14, 15], 
                        "model": [X_series_info] * 8 }
        super().__init__(DEVICENAME, DXL_MODELS, BAUDRATE=115200)

        self.joints_bias = [180.0] * 8
        self.joints_axis = [1,   -1,   1,   1,   1,   -1,   -1,   1]
        self.joints = [0] * 8
        self.joints_increment = [0] * 8
        self.initial_pos = [-20, -45, 2, -20, 0, 40, 92, 0]
        self.limit_high = [ 1.57, 0.12, 1.57, 1.90]
        self.limit_low  = [-1.05,-1.57,-1.57, 0.00]

        self.changeAllMotorOperatingMode(OP_MODE=3)
        self.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[200, 200, 50, 50, 50, 50, 50, 50])
        time.sleep(0.1)

    def to_pose(self, pose, speed=0.5):
        # Get current joint angles.
        current_angles = self.readAllMotorPosition()
        current_angles = [(resolution2degree(current_angles[i])-self.joints_bias[i])*self.joints_axis[i] for i in range(len(current_angles))]

        # Set final joint angles.
        final_angles = current_angles
        if   pose == "initial":   final_angles = self.initial_pos.copy()
        elif pose == "shut down": final_angles = [0]*8

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

        # Get final actual joint angles
        current_angles = self.readAllMotorPosition()
        current_angles = [(resolution2degree(current_angles[i])-self.joints_bias[i])*self.joints_axis[i] for i in range(len(current_angles))]
        self.joints = current_angles.copy()
        time.sleep(0.1)

    def toRolyctrl(self, ctrlpos):
        return [(ctrlpos[i]*self.joints_axis[i]+self.joints_bias[i]) for i in range(len(self.joints_bias))]
    