import numpy as np

def T(DH, i):
    [theta, alpha, a, d] = DH[i]
    # 轉角使用弧度制
    T = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0            ,  np.sin(alpha)              , np.cos(alpha)              , d              ],
        [0            ,  0                          , 0                          , 1              ]
    ])
    return T

def DHtable(joints):
    link0 = [              0.0, np.pi/2,     0.0,     0.0]
    link1 = [np.pi/2+joints[0], np.pi/2,     0.0,  0.2488]
    link2 = [        joints[1],     0.0, -0.1105,     0.0]
    link3 = [np.pi/2+joints[2], np.pi/2,     0.0,     0.0]
    link4 = [np.pi/2+joints[3], np.pi/2,     0.0, -0.1195]
    link5 = [        joints[4], np.pi/2,     0.0,     0.0]
    link6 = [        joints[5], np.pi/2,     0.0,  0.1803]
    return [link0, link1, link2, link3, link4, link5, link6]