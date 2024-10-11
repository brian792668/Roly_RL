import numpy as np
import time
from csv import reader, writer

def resolution2degree(resolution):
    return resolution*360/4096

def degree2resolution(degree):
    return (degree.astype(float)/360*4096).astype(int)

def interpolation(pos_list, interpolate_num):
    interpolation_data = []

    for i in range(len(pos_list)-1):

        start = [float(j) for j in pos_list[i]]
        end = [float(j) for j in pos_list[i+1]]

        i_data = np.linspace(start, end, interpolate_num)
        interpolation_data.append(i_data)

    return interpolation_data

def recordArmPosition(arm, file):
    all_motor_position = arm.getAllMotorPosition()

    write = writer(file)
    write.writerow(all_motor_position)

def playArmPositionSequence(arm, file, velocity = 30):

        read = reader(file)
        pos_list = list(read)

        # # sleep
        # 插值 -> 所有馬達同時轉，同時停，但插值數過少，過程會震
        # 同時送訊號，並控制馬達轉速 -> 所有馬達同時轉，馬達位移較小的會先停
        for TARGET_POSITION in pos_list:
            arm.writeAllMotorProfileVelocity(velocity)
            arm.writeAllMotorPosition(TARGET_POSITION)

            next_point = [float(i) for i in TARGET_POSITION]

            while True:
                current_positions = arm.getAllMotorPosition()
                position_diff = abs(np.array(current_positions) - np.array(next_point))

                # 僅限無負載，若有負重可能須另想辦法
                if np.all(position_diff < 1):
                    print("arrived")
                    break

                time.sleep(0.1)  
