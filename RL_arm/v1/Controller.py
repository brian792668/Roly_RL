class PIDcontroller:
    def __init__(self, parameter, initTarget):
        self.kp = [row[0] for row in parameter]
        self.kv = [row[1] for row in parameter]
        self.ki = [row[2] for row in parameter]
        self.accErr = [0]*len(initTarget)
        self.gain = 1.0

    def getSignal(self, qpos, qvel, target):
        signal = []
        for i in range(len(target)):
            self.accErr[i] += qpos[i] - target[i]
            signal.append((-self.gain*self.kp[i]*(qpos[i] - target[i]) - self.gain*self.ki[i]*self.accErr[i] - self.gain*self.kv[i]*qvel[i])*1.0)
            if signal[i] >= 20:
                signal[i] = 20
            elif signal[i] <= -20:
                signal[i] = -20
        return signal