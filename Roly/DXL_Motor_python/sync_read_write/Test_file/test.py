from test_importlib import *

DEVICENAME = "/dev/ttyUSB0"
DXL_IDS = [1]  
TARGET_POSITIONS = [150]

motor = DXL_Motor(DEVICENAME, DXL_IDS)
motor.checkPortAndBaudRate()

OP_MODE = 3
motor.setAllMotorTorqurDisable()
motor.writeOperatingMode(OP_MODE)
motor.setAllMotorTorqueEnable()
motor.readOperatingMode()
motor.writeAllMotorPosition(TARGET_POSITIONS)
time.sleep(3)
motor.readAllMotorPosition()
motor.setAllMotorTorqurDisable()


try:
    while True:
        motor.readAllMotorPosition()
        time.sleep(0.1) 

except KeyboardInterrupt:
    print("KeyboardInterrupt")

finally:
    motor.setAllMotorTorqurDisable()

motor.portHandler.closePort()
