class Motor_Info():
    def __init__(self):
        '''
        ********* DYNAMIXEL Model definition *********
        ***** (Use only one definition at a time) *****
        MY_DXL = 'X_SERIES'     # X330 (5.0 V recommended), X430, X540, 2X430
        MY_DXL = 'MX_SERIES'    # MX series with 2.0 firmware update.
        MY_DXL = 'PRO_A_SERIES' # PRO series with (A) firmware update.
        MY_DXL = 'P_SERIES'     # PH54, PH42, PM54
        '''
        self.LEN_OPERATING_MODE          = 1         # Data Byte Length
        self.LEN_TORQUE_ENABLE           = 1
        self.LEN_LED_RED                 = 1
        self.LEN_GOAL_POSITION           = 4
        self.LEN_PRESENT_POSITION        = 4
        self.LEN_PROFILE_ACCELERATION    = 4
        self.LEN_PROFILE_VELOCITY        = 4
        self.LEN_PRESENT_VELOCITY        = 4
class X_Motor_Info(Motor_Info):
    def __init__(self):
        self.ADDR_OPERATING_MODE         = 11  
        self.LEN_OPERATING_MODE          = 1
        self.ADDR_TORQUE_ENABLE          = 64
        self.LEN_TORQUE_ENABLE           = 1 
        self.ADDR_LED_RED                = 65
        self.LEN_LED_RED                 = 1         # Data Byte Length
        self.ADDR_GOAL_POSITION          = 116
        self.LEN_GOAL_POSITION           = 4         # Data Byte Length
        self.ADDR_PRESENT_POSITION       = 132
        self.LEN_PRESENT_POSITION        = 4         # Data Byte Length
        self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
        self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
        self.ADDR_PROFILE_ACCELERATION   = 108
        self.LEN_PROFILE_ACCELERATION    = 4
        self.ADDR_PROFILE_VELOCITY       = 112
        self.LEN_PROFILE_VELOCITY        = 4
        self.ADDR_PRESENT_VELOCITY       = 128
        self.LEN_PRESENT_VELOCITY        = 4
class P_Motor_Info(Motor_Info):
    def __init__(self):
        self.ADDR_OPERATING_MODE         = 11  
        self.LEN_OPERATING_MODE          = 1
        self.ADDR_TORQUE_ENABLE          = 512       # Control table address is different in DYNAMIXEL model
        self.LEN_TORQUE_ENABLE           = 1 
        self.ADDR_LED_RED                = 513       # R.G.B Address: 513 (red), 544 (green), 515 (blue)
        self.LEN_LED_RED                 = 1         # Data Byte Length
        self.ADDR_GOAL_POSITION          = 564
        self.LEN_GOAL_POSITION           = 4         # Data Byte Length
        self.ADDR_PRESENT_POSITION       = 580
        self.LEN_PRESENT_POSITION        = 4         # Data Byte Length
        self.DXL_MINIMUM_POSITION_VALUE  = -150000   # Refer to the Minimum Position Limit of product eManual
        self.DXL_MAXIMUM_POSITION_VALUE  = 150000    # Refer to the Maximum Position Limit of product eManual
        self.ADDR_PROFILE_ACCELERATION   = 556
        self.LEN_PROFILE_ACCELERATION    = 4
        self.ADDR_PROFILE_VELOCITY       = 560
        self.LEN_PROFILE_VELOCITY        = 4
        self.ADDR_PRESENT_VELOCITY       = 576
        self.LEN_PRESENT_VELOCITY        = 4