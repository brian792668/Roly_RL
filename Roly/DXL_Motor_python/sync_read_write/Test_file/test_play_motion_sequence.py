from test_importlib import *

DEVICENAME_ARM = "/dev/ttyUSB0"
DXL_IDS_ARM = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24] 
OP_MODE = 4

motion_data = "./UpperBody/Motion_csv_file/data.csv"

file = open(motion_data, mode='r', newline='')

arm = DXL_Motor(DEVICENAME_ARM, DXL_IDS_ARM)
arm.checkPortAndBaudRate()
arm.changeAllMotorOperatingMode(OP_MODE)
arm.readOperatingMode()

playArmPositionSequence(arm, file, velocity=30)

file.close()  
arm.setAllMotorTorqurDisable()

arm.portHandler.closePort()
print("close port")
