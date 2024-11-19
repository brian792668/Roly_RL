import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import time

class Camera():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 60Hz
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30) # 60Hz
        self.pipeline.start(config)  # start pipeline with config
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        self.color_img = np.asanyarray(color_frame.get_data())
        self.depth_img = np.asanyarray(depth_frame.get_data())
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # self.rgbimg = np.zeros((480, 640, 3), dtype=np.uint8)
        # self.depthimg = np.zeros((480, 640), dtype=np.uint8)
        self.target = ['nan', 'nan']
        self.target_depth = float('nan')
        self.track_done = False

    def get_img(self, rgb=True, depth=True):
        frames = self.pipeline.wait_for_frames()
        if rgb == True:
            color_frame = frames.get_color_frame()
            self.color_img = np.asanyarray(color_frame.get_data())
        if depth == True:
            depth_frame = frames.get_depth_frame()
            self.depth_img = np.asanyarray(depth_frame.get_data())
            self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=0.03), cv2.COLORMAP_JET)

    def show(self, rgb = True, depth = True):
        if rgb == True:     cv2.imshow("Realsense D435i RGB", self.color_img)
        if depth == True:   cv2.imshow("Realsense D435i Depth with color", self.depth_colormap)
        cv2.waitKey(1)

    # def get_target(self, depth=False):
    #     # 定義紅色的RGB範圍
    #     lower_red = np.array([100, 0, 0], dtype=np.uint8)
    #     upper_red = np.array([255, 50, 50], dtype=np.uint8)
    #     # 創建紅色遮罩
    #     mask = cv2.inRange(self.rgbimg, lower_red, upper_red)

    #     # 確保在圖像中有紅色物體
    #     if np.any(mask):
    #         # 獲取紅色物體的像素坐標
    #         coords = np.column_stack(np.where(mask > 0))

    #         # 計算紅色物體的中心點
    #         center = np.mean(coords, axis=0)
    #         center_y, center_x = center

    #         # 在RGB圖像上畫十字標
    #         size = 3  # 十字標的大小
    #         thickness = 1  # 線條的粗細

    #         # 畫橫線
    #         cv2.line(self.rgbimg, (int(center_x) - size, int(center_y)), 
    #                  (int(center_x) + size, int(center_y)), (255, 255, 255), thickness)
    #         # 畫縱線
    #         cv2.line(self.rgbimg, (int(center_x), int(center_y) - size), 
    #                  (int(center_x), int(center_y) + size), (255, 255, 255), thickness)

    #         if depth == True:
    #             # 從深度圖像中獲取對應的深度值
    #             self.target_depth = 100*self.depthimg[int(center_y), int(center_x)]

    #             cv2.putText(self.rgbimg, f"{self.target_depth:.1f}", (int(center_x) + 10, int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 
    #                     0.5, (255, 255, 255), 1, cv2.LINE_AA)

    #         # 將像素座標轉換至[-1, 1]區間
    #         norm_x = (center_x / self.rgbimg.shape[1]) * 2 - 1
    #         norm_y = (center_y / self.rgbimg.shape[0]) * 2 - 1
    #         self.target = [norm_x, norm_y]

    #     else:
    #         # 若無紅色物體，返回None並設置target為無效值
    #         self.track_done = False
    #         self.target = [float('nan'), float('nan')]
    #         if depth == True:
    #             self.target_depth = float('nan')
            
    # def track(self, ctrlpos, speed=1.0):
    #     new_pos = ctrlpos.copy()
    #     if np.abs(self.target[0]) <= 0.05 and np.abs(self.target[1]) <= 0.05:
    #         self.track_done = True
    #     else:
    #         self.track_done = False
    #         if np.isnan(self.target[0]) == False:
    #             new_pos[0] += -0.1*self.target[0]*speed
    #             new_pos[1] += -0.1*self.target[1]*speed
    #     return new_pos
    
    def stop(self):
        self.pipeline.stop()

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.head_cam = Camera()
        self.running = True
        self.daemon = True  # 設為 daemon thread，主程序結束時會自動結束
        
    def run(self):
        while self.running:
            self.head_cam.get_img()
            self.head_cam.show(rgb=True, depth=False)
            time.sleep(0.01)  # 小延遲避免 CPU 使用過高
            
    def stop(self):
        self.running = False
        self.head_cam.stop()