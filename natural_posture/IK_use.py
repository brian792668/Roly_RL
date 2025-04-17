import torch
import torch.nn as nn
import torch.nn.functional as F

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(64, 128) # 隱藏層64輸出128
        self.fc3 = nn.Linear(128, 4)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第1層用 ReLU
        x = F.relu(self.fc2(x))  # 第2層用 ReLU
        x = self.fc3(x)          # 最後一層直接輸出
        return x

model = IKMLP()
model.load_state_dict(torch.load('Roly/Inverse_kinematics/models/v7/IKmodel_v7.pth'))
model.eval()  # 切換到評估模式，這樣模型不會更新權重

new_xyz1 = torch.tensor([0.06, -0.46, -0.13], dtype=torch.float32)
new_xyz2 = torch.tensor([0.21, -0.50, -0.13], dtype=torch.float32)
new_xyz3 = torch.tensor([0.25, -0.16, -0.13], dtype=torch.float32)
with torch.no_grad():  # 不需要梯度計算，因為只做推論
    print(model(torch.tensor([0.06, -0.46, -0.13], dtype=torch.float32)).tolist())
    print(model(torch.tensor([0.21, -0.50, -0.13], dtype=torch.float32)).tolist()) 
    print(model(torch.tensor([0.25, -0.16, -0.13], dtype=torch.float32)).tolist())