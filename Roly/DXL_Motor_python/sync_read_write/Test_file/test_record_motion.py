from test_importlib import *

DEVICENAME_ARM = "/dev/ttyUSB0"
DXL_IDS_ARM = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24] 

OP_MODE = 0
motion_data = "./UpperBody/Motion_csv_file/data.csv"
file = open(motion_data, mode='w', newline='')

arm = DXL_Motor(DEVICENAME_ARM, DXL_IDS_ARM)
arm.checkPortAndBaudRate()

print("crtl+c to record motors' positions")

try:
    while True:
        try:
            if OP_MODE != 0:
                OP_MODE = 0
                arm.changeAllMotorOperatingMode(OP_MODE)

            arm.readAllMotorPosition()
            time.sleep(0.5) 

        except KeyboardInterrupt:
            time.sleep(1.5)
            OP_MODE = 4
            arm.changeAllMotorOperatingMode(OP_MODE)

            print("press enter to record position, press q to quit, press n do nothing")
            usr_input = input()

            if usr_input.lower() == 'q':
                print("quit")
                break
            elif usr_input.lower() == 'n':
                print("continue")
            else:
                time.sleep(1)
                recordArmPosition(arm, file)
                time.sleep(0.5)
                print("recorded, and continue")

except Exception as e:
    print(f"error : {e}")

finally:
    file.close()
    arm.setAllMotorTorqurDisable()

    arm.portHandler.closePort()
    print("close port")
