import numpy as np

class DHtable():
    def __init__(self, table):
        self.table = table
    
    def update_hand_length(self, hand_length):
        self.table[-1][3] = 0.17 + hand_length

    def update_camera_distance(self, camera_distance):
        self.table[-1][3] = camera_distance

    def Tans_Matrix(self, link_number, angle):
        [theta, alpha, a, d] = self.table[link_number]
        theta += angle
        # 轉角使用radious
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0            ,  np.sin(alpha)              , np.cos(alpha)              , d              ],
            [0            ,  0                          , 0                          , 1              ]
        ])
        return T
    
    def forward_hand(self, angles, hand_length):
        T01 = self.Tans_Matrix(link_number=0, angle=0)
        T12 = self.Tans_Matrix(link_number=1, angle=angles[0])
        T23 = self.Tans_Matrix(link_number=2, angle=angles[1])
        T34 = self.Tans_Matrix(link_number=3, angle=0)
        T45 = self.Tans_Matrix(link_number=4, angle=angles[2])
        T56 = self.Tans_Matrix(link_number=5, angle=angles[3])
        T6E = self.Tans_Matrix(link_number=6, angle=0)
        T02 = np.dot(T01, T12)
        T03 = np.dot(T02, T23)
        T04 = np.dot(T03, T34)
        T05 = np.dot(T04, T45)
        T06 = np.dot(T05, T56)  
        T0E = np.dot(T06, T6E)
        EE  = np.dot(T0E, np.array([[0], [0], [hand_length], [1]]))
        return [EE[0][0], EE[1][0], EE[2][0]]
    
    def forward_neck(self, angles, camera_distance):
        T01 = self.Tans_Matrix(link_number=0, angle=0)
        T12 = self.Tans_Matrix(link_number=1, angle=angles[0])
        T23 = self.Tans_Matrix(link_number=2, angle=angles[1])
        T02 = np.dot(T01, T12)
        T03 = np.dot(T02, T23)
        EE  = np.dot(T03, np.array([[0], [0], [camera_distance], [1]]))
        return [EE[0][0], EE[1][0], EE[2][0]]
    
    def forward_neck_new(self, angles, target_position):
        T01 = self.Tans_Matrix(link_number=0, angle=0)
        T12 = self.Tans_Matrix(link_number=1, angle=angles[0])
        T23 = self.Tans_Matrix(link_number=2, angle=angles[1])
        T02 = np.dot(T01, T12)
        T03 = np.dot(T02, T23)
        EE  = np.dot(T03, np.array([[target_position[0]], [target_position[1]], [target_position[2]], [1]]))
        return [EE[0][0], EE[1][0], EE[2][0]]