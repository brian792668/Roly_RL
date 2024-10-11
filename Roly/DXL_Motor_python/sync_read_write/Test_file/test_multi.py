from test_importlib import *

DEVICENAME = "/dev/ttyUSB0"
DXL_IDS = [20, 21, 22, 23]  
TARGET_POSITIONS = [0, 180, 0, 0]
OP_MODE = 4

multi_motor = DXL_Motor(DEVICENAME, DXL_IDS)

multi_motor.checkPortAndBaudRate()

multi_motor.changeAllMotorOperatingMode(OP_MODE)

multi_motor.setAllMotorTorqueEnable()

multi_motor.writeAllMotorPosition(TARGET_POSITIONS)
time.sleep(2)
multi_motor.readAllMotorPosition()

multi_motor.setAllMotorTorqurDisable()

multi_motor.portHandler.closePort()
