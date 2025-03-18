import cv2
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
class Camera():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 60Hz
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30) # 60Hz
        self.pipeline.start(self.config)  # start pipeline with config
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        self.color_img = np.asanyarray(color_frame.get_data())
        self.depth_img = np.asanyarray(depth_frame.get_data())
        self.color_mask = self.color_img
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=0.03), cv2.COLORMAP_JET)

        self.target_exist = False
        self.target_pixel = [320, 240]
        self.target_norm = [0.0, 0.0]
        self.target_vel  = [0.0, 0.0]
        self.target_depth = 1.0

        self.colorizer = rs.colorizer()        
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.align = rs.align(rs.stream.color)

        self.pipeline.stop()
        self.is_running = False

        self.hand_center = None
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.9
        )
        self.hand_exist = False
        self.hand_norm = [0.0, 0.0]
        self.hand_vel  = [0.0, 0.0]
        self.hand_depth = 1.0

    def get_img(self, rgb=True, depth=True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        if rgb == True:
            color_frame = frames.get_color_frame()
            # color_frame = aligned_frames.get_color_frame()
            self.color_img = np.asanyarray(color_frame.get_data())
        if depth == True:
            depth_frame = frames.get_depth_frame()
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
            self.depth_img = np.asanyarray(depth_frame.get_data())

            new_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=-0.2), cv2.COLORMAP_JET)
            self.depth_colormap = cv2.addWeighted(self.depth_colormap, 0.1, new_depth_colormap, 0.9, 0)

    def show(self, rgb = True, depth = True):
        if rgb == True:     cv2.imshow("Realsense D435i RGB", self.color_img)
        if depth == True:   cv2.imshow("Realsense D435i Depth with color", self.depth_colormap)
        # if depth == True:   cv2.imshow("Realsense D435i Depth with color", self.depth_img)
        cv2.waitKey(1)

    def get_target(self, depth=False):
        # 定義紅色的RGB範圍
        lower_red = np.array([0, 10, 180], dtype=np.uint8)
        upper_red = np.array([100, 200, 255], dtype=np.uint8)
        # 創建紅色遮罩
        mask = cv2.inRange(self.color_img, lower_red, upper_red)
        self.color_mask = cv2.bitwise_and(self.color_img, self.color_img, mask=mask)
        
        # 將原圖與遮罩層進行混合
        alpha = 0.7  # 透明度
        self.color_img = cv2.addWeighted(self.color_img, alpha, self.color_mask, 1-alpha, 0)

        # 確保在圖像中有紅色物體
        if np.any(mask):
            self.target_exist = True

            # 獲取紅色物體的像素坐標
            coords = np.column_stack(np.where(mask > 0))

            # 計算紅色物體的中心點
            center = np.mean(coords, axis=0)
            center_y, center_x = center

            # 在RGB圖像上畫十字標
            size = 3  # 十字標的大小
            thickness = 1  # 線條的粗細

            # 畫橫線
            cv2.line(self.color_img, (int(center_x) - size, int(center_y)), 
                     (int(center_x) + size, int(center_y)), (255, 255, 255), thickness)
            # 畫縱線
            cv2.line(self.color_img, (int(center_x), int(center_y) - size), 
                     (int(center_x), int(center_y) + size), (255, 255, 255), thickness)

            if depth == True:
                # 從深度圖像中獲取對應的深度值
                self.target_depth = self.depth_img[int(center_y), int(center_x)]*0.001  # m

                cv2.putText(self.color_img, f"{self.target_depth:.3f} m", (int(center_x) + 30, int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 將像素座標轉換至[-1, 1]區間
            norm_x = (center_x / self.color_img.shape[1]) * 2 - 1
            norm_y = (center_y / self.color_img.shape[0]) * 2 - 1

            self.target_vel = [norm_x-self.target_norm[0], norm_y-self.target_norm[1]]
            self.target_norm = [norm_x, norm_y]

        else:
            self.target_exist = False

    def get_hand(self, depth=False):
        color_img = self.color_img
        self.hand_center = None
        img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Mediapipe 需要 RGB 格式
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, handedness in enumerate(results.multi_handedness):
                if handedness.classification[0].label == "Right":  # 只處理左手
                    hand_landmarks = results.multi_hand_landmarks[idx]
                    lmList = []
                    h, w, _ = color_img.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append((cx, cy))

                    # 手部中心點取 5, 17 號關節的平均
                    cx, cy = int((lmList[5][0] + lmList[17][0]) / 2), \
                             int((lmList[5][1] + lmList[17][1]) / 2)
                    self.hand_center = (cx, cy)
                    norm_x = (cx / w) * 2 - 1
                    norm_y = (cy / h) * 2 - 1
                    self.hand_vel = [norm_x - self.hand_norm[0], norm_y - self.hand_norm[1]]
                    self.hand_norm = [norm_x, norm_y]
                    self.hand_exist = True

                    # 在圖片上畫標記
                    cv2.circle(color_img, self.hand_center, 8, (255, 255, 255), -1)
                    for i in [4, 8, 12, 16, 20]:
                        cv2.circle(color_img, lmList[i], 4, (150, 150, 150), -1)

                    self.color_img = color_img

                    if depth:
                        self.hand_depth = self.depth_img[int(cy), int(cx)] * 0.001  # m
                        cv2.putText(self.color_img, f"{self.hand_depth:.3f} m", (int(cx) + 30, int(cy)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    return  # 找到左手就結束
                
        else:
            self.hand_exist = False
        
    def start(self):
        self.pipeline.start(self.config)  # start pipeline with config
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()