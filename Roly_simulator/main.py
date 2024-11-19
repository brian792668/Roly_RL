import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('Roly/RolyURDF2/Roly.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
viewer = mujoco.viewer.launch_passive(model, data, show_right_ui=False)

while viewer.is_running():
    mujoco.mj_step(model, data)
    if int(1000*data.time)%50 == 0: # 50ms rneder 一次
        viewer.sync()
        viewer.cam.azimuth += 0.05 

renderer.close() 