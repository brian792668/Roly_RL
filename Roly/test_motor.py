import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info
from imports.Camera import *


X_series_info = X_Motor_Info()
P_series_info = P_Motor_Info()

DEVICENAME = "/dev/ttyUSB0"
DXL_MODELS = {"id":[1, 2, 10, 11, 12, 13, 14, 15], "model":[X_series_info, X_series_info, X_series_info, X_series_info, X_series_info, X_series_info, X_series_info, X_series_info]}

motor = DXL_Motor(DEVICENAME, DXL_MODELS, BAUDRATE=115200)
motor.changeAllMotorOperatingMode(OP_MODE=3)
motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[10]*len(motor.pos_ctrl))
motor.setAllMotorTorqueEnable()
time.sleep(0.1) 

# head_cam = Camera()
# time.sleep(5) 

if __name__ == '__main__':
    for i in range(150):
        motor.pos_ctrl = motor.toRolyctrl([0, 0, 0, -20, 0, 0, 0, 0])
        motor.writeAllMotorPosition(motor.pos_ctrl)
        # head_cam.get_img()
        # head_cam.show(rgb=True, depth=False)
    # for i in range(50):
    #     motor.move([0, 0, 0, -20, 0, 0, 0, 0], speed=0.2)
    #     head_cam.get_img()
    #     head_cam.show(rgb=True, depth=False)

    motor.vel = [1]*len(motor.pos_ctrl)
    motor.writeAllMotorProfileVelocity(motor.vel)
    for i in range(100):
        motor.move([-30, 20, 60, -60, 60, -60, -60, 0], speed=0.9)
        # head_cam.get_img()
        # head_cam.show(rgb=True, depth=False)
    motor.vel = [1]*len(motor.pos_ctrl)
    motor.writeAllMotorProfileVelocity(motor.vel)
    for i in range(200):
        motor.move([0, 0, 0, -20, 0, 0, 0, 0], speed=0.2)
        # head_cam.get_img()
        # head_cam.show(rgb=True, depth=False)
        
motor.setAllMotorTorqurDisable()
time.sleep(0.1) 
motor.portHandler.closePort()
# head_cam.stop()