from test_importlib import *

DEVICENAME = "/dev/ttyUSB0"
DXL_IDS = [10]  

motor = DXL_Motor(DEVICENAME, DXL_IDS)
motor.checkPortAndBaudRate()
motor.setAllMotorTorqueEnable()

try:
    while True:
        motor.readAllMotorPosition()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("KeyboardInterrupt")

finally:
    motor.setAllMotorTorqurDisable(DXL_IDS)
    print("toeque disable")

    motor.portHandler.closePort()
    print("close port!")