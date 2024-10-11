'''
only for X-series motor.
change the address table for other series motors. 
'''
from python.src.dynamixel_sdk import *
from helper_function import *

BAUDRATE = 57600 

ADDR_OPERATING_MODE = 11  
ADDR_TORQUE_ENABLE = 64  
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_CURRENT = 126
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY  = 112
ADDR_GOAL_POSITION = 116  
ADDR_PRESENT_POSITION = 132  

SIZE_OPERATING_MODE = 1
SIZE_TORQUE_ENABLE = 1 
SIZE_GOAL_CURRENT = 2
SIZE_PRESENT_CURRENT = 2
SIZE_PROFILE_ACCELERATION = 4
SIZE_PROFILE_VELOCITY  = 4
SIZE_GOAL_POSITION = 4
SIZE_PRESENT_POSITION = 4

packetHandler = PacketHandler(2.0) 

class DXL_Motor():
    def __init__(self, DEVICENAME, DXL_IDS):
        self.DEVICENAME = DEVICENAME
        self.DXL_IDS = DXL_IDS

        self.portHandler = PortHandler(self.DEVICENAME)

    def checkPortAndBaudRate(self):
        if not self.portHandler.openPort():
            print("error opening port")
            quit()

        if not self.portHandler.setBaudRate(BAUDRATE):
            print("error setting buadrate")
            quit()

    def readAllMotorStatus(self, address, data_size):
        groupSyncRead = GroupSyncRead(self.portHandler, packetHandler, address, data_size)

        for dxl_id in self.DXL_IDS:
            groupSyncRead.addParam(dxl_id)

        dxl_comm_result = groupSyncRead.txRxPacket()

        motor_data = []
        for dxl_id in self.DXL_IDS:
            motor_data.append(groupSyncRead.getData(dxl_id, address, data_size))

        return motor_data, dxl_comm_result
    
    def writeAllMotorStatus(self, address, target_values, data_size):
        groupSyncWrite = GroupSyncWrite(self.portHandler, packetHandler, address, data_size)

        for i, dxl_id in enumerate(self.DXL_IDS):
            if data_size == 4:
                data = [DXL_LOBYTE(DXL_LOWORD(target_values[i])),
                        DXL_HIBYTE(DXL_LOWORD(target_values[i])),
                        DXL_LOBYTE(DXL_HIWORD(target_values[i])),
                        DXL_HIBYTE(DXL_HIWORD(target_values[i]))]
            elif data_size == 2:
                data = [DXL_LOBYTE(target_values[i]), DXL_HIBYTE(target_values[i])]
            else:
                data = [target_values[i]]
        
            groupSyncWrite.addParam(dxl_id, data)

        dxl_comm_result = groupSyncWrite.txPacket()

        return dxl_comm_result

    def readOperatingMode(self):
        motor_data, dxl_comm_result = self.readAllMotorStatus(ADDR_OPERATING_MODE, SIZE_OPERATING_MODE)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Read operation mode fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully read operation mode : {motor_data[0]}")

    def writeOperatingMode(self, OP_MODE):
        dxl_comm_result = self.writeAllMotorStatus(ADDR_OPERATING_MODE, [OP_MODE]*len(self.DXL_IDS), SIZE_OPERATING_MODE)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write operation mode fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully set operation mode to : {OP_MODE}")

    def setAllMotorTorqueEnable(self):
        dxl_comm_result = self.writeAllMotorStatus(ADDR_TORQUE_ENABLE, [1]*len(self.DXL_IDS), SIZE_TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Enable motor torque fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Successfully enable all motor torque")

    def setAllMotorTorqurDisable(self):   
        dxl_comm_result = self.writeAllMotorStatus(ADDR_TORQUE_ENABLE, [0]*len(self.DXL_IDS), SIZE_TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Disable motor torque fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Successfully disable all motor torque")

    def writeAllMotorPosition(self, TARGET_POSITIONS):
        RESOLUSION = degree2resolution(np.array(TARGET_POSITIONS))
        dxl_comm_result = self.writeAllMotorStatus(ADDR_GOAL_POSITION, RESOLUSION, SIZE_GOAL_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor position fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Successfully write all motor position")

    def readAllMotorPosition(self):
        present_resolution, dxl_comm_result = self.readAllMotorStatus(ADDR_PRESENT_POSITION, SIZE_PRESENT_POSITION)
        present_degree = [resolution2degree(present_resolution[i]) for i in range(len(present_resolution))]
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Read motor position fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            for i, dxl_id in enumerate(self.DXL_IDS):
                print(f"motor {dxl_id}'s position is {present_degree[i]}") # 關

    def writeAllMotorProfileVelocity(self, PROFILE_VELOCITY):
        dxl_comm_result = self.writeAllMotorStatus(ADDR_PROFILE_VELOCITY , [PROFILE_VELOCITY]*len(self.DXL_IDS), SIZE_PROFILE_VELOCITY)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor profile velocity fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully write all motor profile velocity to {PROFILE_VELOCITY}") # 關

    def writeAllMotorProfileAcceleration(self, PROFILE_ACCELERATION):
        dxl_comm_result = self.writeAllMotorStatus(ADDR_PROFILE_ACCELERATION, [PROFILE_ACCELERATION]*len(self.DXL_IDS), SIZE_PROFILE_ACCELERATION)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor profile acceleration fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully write all motor profile acceleration to {PROFILE_ACCELERATION}") # 關

    def writeAllMotorCurrent(self, TARGET_CURRENT):
        dxl_comm_result = self.writeAllMotorStatus(ADDR_PROFILE_ACCELERATION, [TARGET_CURRENT]*len(self.DXL_IDS), SIZE_PROFILE_ACCELERATION)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor current fail : {packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully write all motor current to {TARGET_CURRENT}") # 關

    def changeAllMotorOperatingMode(self, OP_MODE):
        self.setAllMotorTorqurDisable()
        time.sleep(0.01)
        self.writeOperatingMode(OP_MODE)
        time.sleep(0.01)
        self.setAllMotorTorqueEnable()
        time.sleep(0.01)

    def getAllMotorPosition(self):
        present_resolution, _ = self.readAllMotorStatus(ADDR_PRESENT_POSITION, SIZE_PRESENT_POSITION)
        present_degree = [resolution2degree(present_resolution[i]) for i in range(len(present_resolution))]

        return present_degree
    
    def getAllMotorCurrent(self):
        present_current, _ = self.readAllMotorStatus(ADDR_PRESENT_CURRENT, SIZE_PRESENT_CURRENT)

        return present_current