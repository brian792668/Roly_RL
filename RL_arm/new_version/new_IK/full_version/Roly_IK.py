import mujoco
import mujoco.viewer
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports.Controller import *
from imports.Forward import *
from imports.info import *
from imports.Settings import *

class IKMLP(nn.Module):
    def __init__(self): # 3 -> 32 -> 16 -> 2
        super(IKMLP, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class IKDataset(Dataset):
    def __init__(self, IK_data):
        self.IK_data = IK_data
    
    def __len__(self):
        return len(self.IK_data)
    
    def __getitem__(self, idx):
        IK_data = self.IK_data[idx]
        IK_label_tensor = torch.tensor(IK_data, dtype=torch.float32)
        return IK_label_tensor

class IK():
    def __init__(self):
        self.robot = mujoco.MjModel.from_xml_path('Roly/Roly_XML/Roly.xml')
        self.data = mujoco.MjData(self.robot)
        self.renderer = mujoco.Renderer(self.robot)

        self.speed = 0.1
        self.timestep = 0
        self.info = info()
    
    def step(self):
        for i in range(int(1/self.info.Hz/0.005)):
            self.info.ctrlpos[2] = self.info.ctrlpos[2] + np.tanh(1.2*self.info.arm_target_pos[0] - 1.2*self.info.pos[2])*0.01
            self.info.ctrlpos[3] = self.info.ctrlpos[3] + np.tanh(1.2*self.info.arm_target_pos[1] - 1.2*self.info.pos[3])*0.01
            self.info.ctrlpos[4] = 0
            self.info.ctrlpos[5] = self.info.ctrlpos[5] + np.tanh(1.2*self.info.arm_target_pos[3] - 1.2*self.info.pos[5])*0.01
            self.info.ctrlpos[6] = self.info.ctrlpos[6] + np.tanh(1.2*self.info.arm_target_pos[4] - 1.2*self.info.pos[6])*0.01
            if   self.info.ctrlpos[2] > self.info.limit_high[0]: self.info.ctrlpos[2] = self.info.limit_high[0]
            elif self.info.ctrlpos[2] < self.info.limit_low[0] : self.info.ctrlpos[2] = self.info.limit_low[0]
            if   self.info.ctrlpos[3] > self.info.limit_high[1]: self.info.ctrlpos[3] = self.info.limit_high[1]
            elif self.info.ctrlpos[3] < self.info.limit_low[1] : self.info.ctrlpos[3] = self.info.limit_low[1]
            if   self.info.ctrlpos[5] > self.info.limit_high[2]: self.info.ctrlpos[5] = self.info.limit_high[2]
            elif self.info.ctrlpos[5] < self.info.limit_low[2] : self.info.ctrlpos[5] = self.info.limit_low[2]
            if   self.info.ctrlpos[6] > self.info.limit_high[3]: self.info.ctrlpos[6] = self.info.limit_high[3]
            elif self.info.ctrlpos[6] < self.info.limit_low[3] : self.info.ctrlpos[6] = self.info.limit_low[3]

            self.info.pos = [self.data.qpos[i] for i in controlList]
            self.info.vel = [self.data.qvel[i-1] for i in controlList]
            self.data.ctrl[:] = self.info.PIDctrl.getSignal(self.info.pos, self.info.vel, self.info.ctrlpos)
            mujoco.mj_step(self.robot, self.data)

        if self.timestep%int(49*self.speed+1) == 0:
            self.viewer.sync()

    def label(self, numberofpoints=50000, batch_points=10000, speed = 1.0):
        self.viewer = mujoco.viewer.launch_passive(self.robot, self.data, show_right_ui= False)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0.3, 0.0, 1.0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.azimuth = 200
        for i in range(50):
            self.step()

        self.speed = speed
        self.info.pos_hand   = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")]
        self.info.pos_origin = self.data.site_xpos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"origin_marker")]
        file_path = os.path.dirname(os.path.abspath(__file__))
        label = np.full((batch_points, 8), np.nan, dtype=np.float32)
        number_of_point = 0
        number_of_label = 0
        while self.viewer.is_running():
            # test data
            for i in range(10):
                if self.timestep%int(2*self.info.Hz) == 0:
                    self.info.arm_target_pos[0] = random.uniform(self.info.limit_low[0], self.info.limit_high[0])
                    self.info.arm_target_pos[1] = random.uniform(self.info.limit_low[1], self.info.limit_high[1])
                    self.info.arm_target_pos[3] = random.uniform(self.info.limit_low[2], self.info.limit_high[2])
                    self.info.arm_target_pos[4] = random.uniform(self.info.limit_low[3], self.info.limit_high[3])
                self.step()
                self.timestep += 1

            # # train data
            # if self.timestep%int(2*self.info.Hz) == 0:
            #     self.info.arm_target_pos[0] = random.uniform(self.info.limit_low[0], self.info.limit_high[0])
            #     self.info.arm_target_pos[1] = random.uniform(self.info.limit_low[1], self.info.limit_high[1])
            #     self.info.arm_target_pos[3] = random.uniform(self.info.limit_low[2], self.info.limit_high[2])
            #     self.info.arm_target_pos[4] = random.uniform(self.info.limit_low[3], self.info.limit_high[3])
            # self.step()
            # self.timestep += 1

            # label 
            self.info.joints = [self.info.pos[2], self.info.pos[3], self.info.pos[5], self.info.pos[6]]
            for i in range(5):
                self.info.hand_length = random.uniform(0.0, 0.10)
                self.robot.site_pos[mujoco.mj_name2id(self.robot, mujoco.mjtObj.mjOBJ_SITE, f"R_hand_marker")][2] = 0.17 + self.info.hand_length
                mujoco.mj_forward(self.robot, self.data)
                label[number_of_point+i] = [
                    self.info.pos_hand[0] - self.info.pos_origin[0], 
                    self.info.pos_hand[1] - self.info.pos_origin[1], 
                    self.info.pos_hand[2] - self.info.pos_origin[2], 
                    self.info.hand_length,
                    self.info.joints[0], 
                    self.info.joints[1], 
                    self.info.joints[2], 
                    self.info.joints[3] 
                ]
            number_of_point += 5

            # save label
            if number_of_point >= batch_points:
                folder_path = os.path.join(file_path, f"datasets/new/{numberofpoints}_points")
                os.makedirs(folder_path, exist_ok=True)
                np.save(os.path.join(folder_path, f"IK_label_{batch_points}points_{number_of_label}.npy"), label)
                number_of_point = 0
                number_of_label += 1
                print(f"label shape: {label.shape}  number_of_label = {number_of_label}/{int(numberofpoints/batch_points)}  total points: {number_of_label*label.shape[0]}")
                if number_of_label >= int(numberofpoints/batch_points):
                    self.renderer.close() 
                    self.viewer.close()

        self.renderer.close() 
        self.viewer.close()

    def merge(self, numberofpoints, batch_points):
        file_path = os.path.dirname(os.path.abspath(__file__))
        input_directory = os.path.join(file_path, f"datasets/new/{numberofpoints}_points")
        output_directory = os.path.join(file_path, f"datasets/{numberofpoints}points")
        os.makedirs(output_directory, exist_ok=True)
        merged_array = np.array([])  # 預先分配記憶體

        for i in range(int(numberofpoints/batch_points)):
            file_path = os.path.join(input_directory, f"IK_label_{batch_points}points_{i}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)  # 讀取 .npy 檔案

                if merged_array.size == 0:
                    merged_array = data.copy()
                else:
                    merged_array = np.concatenate((merged_array, data.copy()), axis=0)

            else:
                raise FileNotFoundError(f"File not found: {file_path}")


        # 儲存合併後的檔案
        print("shape = ", merged_array.shape)
        output_file = os.path.join(output_directory, f"IK_label_{numberofpoints}points.npy")
        np.save(output_file, merged_array)

    def train(self, train_points=4000000, test_points=4000):
        file_path = os.path.dirname(os.path.abspath(__file__))
        # ---------------- Create datasets -----------------
        IK_label_array = np.load(os.path.join(file_path, f"datasets/{train_points}points/IK_label_{train_points}points.npy"))
        ik_dataset = IKDataset(IK_label_array)
        dataloader = DataLoader(ik_dataset, batch_size=1024, shuffle=True)
        # ---------------- Create testing datasets -----------------
        test_IK_label_array = np.load(os.path.join(file_path, f"datasets/{test_points}points/IK_label_{test_points}points.npy"))
        test_dataset = IKDataset(test_IK_label_array)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # ----------------------- Model --------------------------
        model = IKMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        # --------------------------- Training ------------------------------
        epochs = 100
        train_losses = []
        test_losses = []
        min_train_loss = 1000

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for batch in dataloader:
                # 準備輸入資料：取第 0,1,2,3,6 欄位
                input_batch = torch.cat((
                    batch[:, 0:4],           # index 0,1,2,3
                    batch[:, 6].unsqueeze(1) # index 6
                ), dim=1)

                # 預測模型輸出 [a, b, d]
                predicted = model(input_batch)

                # Ground truth：index 4, 5, 7
                gt_output = torch.cat((
                    batch[:, 4].unsqueeze(1),
                    batch[:, 5].unsqueeze(1),
                    batch[:, 7].unsqueeze(1)
                ), dim=1)

                # 計算 loss
                loss = criterion(predicted, gt_output)

                # 清空梯度
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累積當前 batch 的 loss
                running_loss += loss.item()


            # 計算訓練的平均 loss
            epoch_loss = running_loss / len(dataloader)
            train_losses.append(epoch_loss)

            # ---------------------- Testing ----------------------
            model.eval()
            test_running_loss = 0.0
            with torch.no_grad():
                for batch in test_dataloader:
                    # 準備輸入資料：取第 0,1,2,3,6 欄位
                    input_batch = torch.cat((
                        batch[:, 0:4],           # index 0,1,2,3
                        batch[:, 6].unsqueeze(1) # index 6
                    ), dim=1)

                    # 預測模型輸出 [a, b, d]
                    predicted = model(input_batch)

                    # Ground truth：index 4, 5, 7
                    gt_output = torch.cat((
                        batch[:, 4].unsqueeze(1),
                        batch[:, 5].unsqueeze(1),
                        batch[:, 7].unsqueeze(1)
                    ), dim=1)

                    # 計算 loss
                    loss = criterion(predicted, gt_output)

                    # 累積當前 batch 的 loss
                    test_running_loss += loss.item()

            test_epoch_loss = test_running_loss / len(test_dataloader)
            test_losses.append(test_epoch_loss)

            # 儲存最小的 train loss
            if epoch_loss < min_train_loss:
                min_train_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(file_path, f'IKMLP.pth'))

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}')

            # 每個 epoch 繪製並儲存圖
            plt.figure(figsize=(12, 8))
            plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
            plt.plot(range(1, epoch+2), test_losses, label='Testing Loss', linestyle='dashed')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training & Testing Loss vs. Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(file_path, f"IKMLP.png"))
            plt.close()

    def use(self):
        pass


if __name__ == '__main__':
    Roly_IK = IK()
    # Roly_IK.label(numberofpoints=4000000, batch_points=10000, speed=2.0)
    # Roly_IK.merge(numberofpoints=4000000, batch_points=10000)
    Roly_IK.train(train_points=4000000, test_points=4000)