import numpy as np

def IK(point, theta3):
    theta = [0, 0, 0, 0]
    d1 = 0.3708
    d2 = 0.17
    d3 = (point[0]**2 + point[1]**2 + point[2]**2 )**0.5
    theta[4-1] = np.pi - np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))

    # print("IK", np.degrees(theta[4-1]))
    return(theta.copy())


# IK([0.0, 0.0, -0.3], 0)