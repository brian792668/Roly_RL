import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DXL_Motor_python.bulk_read_write.func.dynamixel_bulk import *
from DXL_Motor_python.bulk_read_write.func.motor_info import X_Motor_Info, P_Motor_Info
from imports.Camera import *

def init_motor():
    X_series_info = X_Motor_Info()
    P_series_info = P_Motor_Info()

    DEVICENAME = "/dev/ttyUSB0"
    DXL_MODELS = {
        "id": [1, 2, 10, 11, 12, 13, 14, 15], 
        "model": [X_series_info] * 8
    }

    motor = DXL_Motor(DEVICENAME, DXL_MODELS, BAUDRATE=115200)
    motor.changeAllMotorOperatingMode(OP_MODE=3)
    motor.writeAllMotorProfileVelocity(PROFILE_VELOCITY=[10]*len(motor.pos_ctrl))
    motor.setAllMotorTorqueEnable()
    time.sleep(0.1)
    return motor

if __name__ == '__main__':

    motor = init_motor()
    camera_thread = CameraThread()
    camera_thread.start()

    try:
        time0 = time.time()
        # 第一段動作
        for i in range(150):
            motor.pos_ctrl = motor.toRolyctrl([0, 0, 0, -20, 0, 0, 0, 0])
            motor.writeAllMotorPosition(motor.pos_ctrl)
            # time.sleep(0.01)
        time1 = time.time()
        print(f"Hz = {150.0/(time1-time0)}")
        time0 = time.time()

        # 第二段動作
        motor.vel = [1]*len(motor.pos_ctrl)
        motor.writeAllMotorProfileVelocity(motor.vel)
        for i in range(100):
            motor.move([-30, 20, 60, -60, 60, -60, -60, 0], speed=0.9)
            # time.sleep(0.01)
        time1 = time.time()
        print(f"Hz = {100.0/(time1-time0)}")
        time0 = time.time()

        # 第三段動作
        motor.writeAllMotorProfileVelocity(motor.vel)
        for i in range(200):
            motor.move([0, 0, 0, -20, 0, 0, 0, 0], speed=0.2)
            # time.sleep(0.01)
        time1 = time.time()
        print(f"Hz = {200.0/(time1-time0)}")
        time0 = time.time()

    finally:
        # 清理資源
        camera_thread.stop()
        camera_thread.join()  # 等待相機執行緒結束
        motor.setAllMotorTorqurDisable()
        time.sleep(0.1)
        motor.portHandler.closePort()