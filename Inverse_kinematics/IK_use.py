import torch
import torch.nn as nn
import torch.nn.functional as F

class IKMLP(nn.Module):
    def __init__(self):
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # 輸入3維，隱藏層64
        self.fc2 = nn.Linear(128, 128) # 隱藏層64輸出128
        self.fc3 = nn.Linear(128, 4)  # 輸出4維（4個關節角度）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一層用 ReLU
        x = F.relu(self.fc2(x))  # 第二層用 ReLU
        x = self.fc3(x)          # 最後一層直接輸出
        return x

model = IKMLP()
model.load_state_dict(torch.load('Roly/Inverse_kinematics/Roly_IK_model.pth'))
model.eval()  # 切換到評估模式，這樣模型不會更新權重

new_xyz = torch.tensor([0.14, -0.58, -0.10], dtype=torch.float32)
with torch.no_grad():  # 不需要梯度計算，因為只做推論
    predicted_angles = model(new_xyz)
    print("\n", predicted_angles)