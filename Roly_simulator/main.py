import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('Roly/RolyURDF2/Roly.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
viewer = mujoco.viewer.launch_passive(model, data, show_right_ui=False)
    
# robot 是模型
# data 是模擬環境負責紀錄各項物理參數的地方
# renderer 會負責處理渲染
# viewer 是觀看的視窗

while viewer.is_running():
    mujoco.mj_step(model, data) # 環境迭代一個timestep
    viewer.sync()               # 會render一次並顯示在視窗中
    viewer.cam.azimuth += 0.05  # 環繞模擬環境（比較帥？）