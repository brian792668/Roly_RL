from .python.src.dynamixel_sdk import *
from .helper_function import *

'''
Notice: Only support protocol 2.0 motors
'''

class DXL_Motor():
    def __init__(self, DEVICENAME, DXL_MODELS, BAUDRATE):
        self.DEVICENAME = DEVICENAME
        self.DXL_MODELS = DXL_MODELS

        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(2.0)
        self.checkPortAndBaudRate(BAUDRATE)

    def checkPortAndBaudRate(self, BAUDRATE=115200):
        if not self.portHandler.openPort():
            print("error opening port")
            quit()

        if not self.portHandler.setBaudRate(BAUDRATE):
            print("error setting buadrate")
            quit()

    def getMotorInfo(self, idx, mode):
        if mode == "OPMODE":
            return self.DXL_MODELS["model"][idx].ADDR_OPERATING_MODE, self.DXL_MODELS["model"][idx].LEN_OPERATING_MODE
        elif mode == "TORQUE_ENABLE":
            return self.DXL_MODELS["model"][idx].ADDR_TORQUE_ENABLE, self.DXL_MODELS["model"][idx].LEN_TORQUE_ENABLE
        elif mode == "LED_RED":
            return self.DXL_MODELS["model"][idx].ADDR_LED_RED, self.DXL_MODELS["model"][idx].LEN_LED_RED
        elif mode == "GOAL_POSITION":
            return self.DXL_MODELS["model"][idx].ADDR_GOAL_POSITION, self.DXL_MODELS["model"][idx].LEN_GOAL_POSITION
        elif mode == "PRESENT_POSITION":
            return self.DXL_MODELS["model"][idx].ADDR_PRESENT_POSITION, self.DXL_MODELS["model"][idx].LEN_PRESENT_POSITION
        elif mode == "PROFILE_ACCELERATION":
            return self.DXL_MODELS["model"][idx].ADDR_PROFILE_ACCELERATION, self.DXL_MODELS["model"][idx].LEN_PROFILE_ACCELERATION
        elif mode == "PROFILE_VELOCITY":
            return self.DXL_MODELS["model"][idx].ADDR_PROFILE_VELOCITY, self.DXL_MODELS["model"][idx].LEN_PROFILE_VELOCITY
        elif mode == "PRESENT_VELOCITY":
            return self.DXL_MODELS["model"][idx].ADDR_PRESENT_VELOCITY, self.DXL_MODELS["model"][idx].LEN_PRESENT_VELOCITY
        
    def readAllMotorStatus(self, mode):
        groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)

        for idx, dxl_id in enumerate(self.DXL_MODELS["id"]):
            address, data_size = self.getMotorInfo(idx, mode)
            groupBulkRead.addParam(dxl_id, address, data_size)

        dxl_comm_result = groupBulkRead.txRxPacket()

        motor_data = []
        for idx, dxl_id in enumerate(self.DXL_MODELS["id"]):
            address, data_size = self.getMotorInfo(idx, mode)
            motor_data.append(groupBulkRead.getData(dxl_id, address, data_size))

        return motor_data, dxl_comm_result
    
    def writeAllMotorStatus(self, target_values, mode):
        groupBulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)

        for idx, dxl_id in enumerate(self.DXL_MODELS["id"]):
            address, data_size = self.getMotorInfo(idx, mode)
            if data_size == 4:
                data = [DXL_LOBYTE(DXL_LOWORD(target_values[idx])),
                        DXL_HIBYTE(DXL_LOWORD(target_values[idx])),
                        DXL_LOBYTE(DXL_HIWORD(target_values[idx])),
                        DXL_HIBYTE(DXL_HIWORD(target_values[idx]))]
            elif data_size == 2:
                data = [DXL_LOBYTE(target_values[idx]), DXL_HIBYTE(target_values[idx])]
            else:
                data = [target_values[idx]]
        
            groupBulkWrite.addParam(dxl_id, address, data_size, data)

        dxl_comm_result = groupBulkWrite.txPacket()

        return dxl_comm_result

    def readOperatingMode(self):
        motor_data, dxl_comm_result = self.readAllMotorStatus("OPMODE")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Read operation mode fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully read operation mode : {motor_data[0]}")

    def writeOperatingMode(self, OP_MODE):
        dxl_comm_result = self.writeAllMotorStatus([OP_MODE]*len(self.DXL_MODELS["id"]), "OPMODE")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write operation mode fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully set operation mode to : {OP_MODE}")

    def setAllMotorTorqueEnable(self):
        dxl_comm_result = self.writeAllMotorStatus([1]*len(self.DXL_MODELS["id"]), "TORQUE_ENABLE")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Enable motor torque fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Successfully enable all motor torque")

    def setAllMotorTorqurDisable(self):   
        dxl_comm_result = self.writeAllMotorStatus([0]*len(self.DXL_MODELS["id"]), "TORQUE_ENABLE")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Disable motor torque fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Successfully disable all motor torque")

    def readAllMotorPosition(self):
        present_resolution, dxl_comm_result = self.readAllMotorStatus("PRESENT_POSITION")
        
        if dxl_comm_result != COMM_SUCCESS:
            pass
            print(f"Read motor position fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            # return pos_read
            return present_resolution
            # for i, dxl_id in enumerate(self.DXL_MODELS["id"]):
            #     print(f"motor {dxl_id}'s position is {present_degree[i]}") # 關

    def writeAllMotorPosition(self, TARGET_POSITIONS):
        RESOLUSION = degree2resolution(np.clip(np.array(TARGET_POSITIONS), 0.1, 359.9))
        dxl_comm_result = self.writeAllMotorStatus(RESOLUSION, "GOAL_POSITION")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor position fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        # else:
        #     print("Successfully write all motor position")

    def readAllMotorProfileVelocity(self):
        profile_velocity, dxl_comm_result = self.readAllMotorStatus("PROFILE_VELOCITY")
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Read motor position fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            for i, dxl_id in enumerate(self.DXL_MODELS["id"]):
                print(f"motor {dxl_id}'s profile velocity is {profile_velocity[i]}") # 關

    def writeAllMotorProfileVelocity(self, PROFILE_VELOCITY):
        dxl_comm_result = self.writeAllMotorStatus(PROFILE_VELOCITY, "PROFILE_VELOCITY")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor profile velocity fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully write all motor profile velocity to {PROFILE_VELOCITY}") # 關

    def writeAllMotorProfileAcceleration(self, PROFILE_ACCELERATION):
        dxl_comm_result = self.writeAllMotorStatus(PROFILE_ACCELERATION, "PROFILE_ACCELERATION")
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Write motor profile acceleration fail : {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print(f"Successfully write all motor profile acceleration to {PROFILE_ACCELERATION}") # 關

    def changeAllMotorOperatingMode(self, OP_MODE):
        self.setAllMotorTorqurDisable()
        self.writeOperatingMode(OP_MODE)
        self.setAllMotorTorqueEnable()
