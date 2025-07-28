import torch
import torch.nn as nn
import torch.nn.functional as F

class NPMLP(nn.Module):
    def __init__(self):
        super(NPMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第1層用 ReLU
        x = F.relu(self.fc2(x))  # 第2層用 ReLU
        x = F.relu(self.fc3(x))  # 第2層用 ReLU
        x = self.fc4(x)          # 最後一層直接輸出
        return x

class CBMLP(nn.Module):
    def __init__(self): # 3 -> 32 -> 16 -> 2
        super(CBMLP, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) * 1.6
        return x
    
class NPandCB(nn.Module):
    def __init__(self, net1, net2):
        super(NPandCB, self).__init__()
        self.NPnet = net1
        self.CBnet = net2

    def forward(self, x):
        natural_posture = self.NPnet(x)  # shape: [batch_size, 1]
        collision_bound = self.CBnet(x)  # shape: [batch_size, 2]
        return torch.cat((natural_posture, collision_bound), dim=1)  # shape: [batch_size, 3]