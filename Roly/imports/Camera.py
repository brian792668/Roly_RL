import cv2
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp

class Camera():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        self.image_size = (160, 120)

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        self.color_img = cv2.resize(np.asanyarray(color_frame.get_data()), self.image_size, interpolation=cv2.INTER_AREA)
        self.depth_img = cv2.resize(np.asanyarray(depth_frame.get_data()), self.image_size, interpolation=cv2.INTER_AREA)
        self.color_mask = self.color_img
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=0.03), cv2.COLORMAP_JET)

        self.target_exist = False
        self.target_pixel = [320, 240]
        self.target_norm = [0.0, 0.0]
        self.target_vel = [0.0, 0.0]
        self.target_depth = 1.0
        self.target_position = [0.0, 0.0, 0.0]
        self.target_vis_data = None  # 用來記錄畫圖資訊

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
            min_detection_confidence=0.7
        )
        self.hand_exist = False
        self.hand_norm = [0.0, 0.0]
        self.hand_vel = [0.0, 0.0]
        self.hand_depth = 1.0
        self.hand_vis_data = None  # 用來記錄畫圖資訊

    def get_img(self, rgb=True, depth=True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        if rgb:
            color_frame = frames.get_color_frame()
            self.color_img = cv2.resize(np.asanyarray(color_frame.get_data()), self.image_size, interpolation=cv2.INTER_AREA)
            self.color_img = cv2.rotate(self.color_img, cv2.ROTATE_180)
        if depth:
            depth_frame = frames.get_depth_frame()
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
            self.depth_img = cv2.resize(np.asanyarray(depth_frame.get_data()), self.image_size, interpolation=cv2.INTER_AREA)
            self.depth_img = cv2.rotate(self.depth_img, cv2.ROTATE_180)

            new_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=-0.2), cv2.COLORMAP_JET)
            self.depth_colormap = cv2.addWeighted(self.depth_colormap, 0.1, new_depth_colormap, 0.9, 0)

    def show(self, rgb=True, depth=True):
        show_image_size = (1000, 750)
        if rgb:
            img = cv2.resize(self.color_img, show_image_size, interpolation=cv2.INTER_AREA)

            if self.target_vis_data:
                mean_x, mean_y, depth_text, position_text = self.target_vis_data
                scale_x = show_image_size[0] / self.image_size[0]
                scale_y = show_image_size[1] / self.image_size[1]
                mx, my = int(mean_x * scale_x), int(mean_y * scale_y)
                cv2.line(img, (mx - 3, my), (mx + 3, my), (255, 255, 255), 1)
                cv2.line(img, (mx, my - 3), (mx, my + 3), (255, 255, 255), 1)
                cv2.putText(img, depth_text, (mx + 30, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, position_text, (mx + 30, my + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if self.hand_vis_data:
                cx, cy, depth_text, position_text = self.hand_vis_data
                scale_x = show_image_size[0] / self.image_size[0]
                scale_y = show_image_size[1] / self.image_size[1]
                hx, hy = int(cx * scale_x), int(cy * scale_y)
                cv2.circle(img, (hx, hy), 8, (255, 255, 255), -1)
                cv2.putText(img, depth_text, (hx + 30, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, position_text, (hx + 30, hy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Realsense D435i RGB", img)

        if depth:
            cv2.imshow("Realsense D435i Depth", cv2.resize(self.depth_colormap, (640, 480), interpolation=cv2.INTER_AREA))

        cv2.waitKey(1)

    def get_target(self):
        self.hand_vis_data = None
        self.target_vis_data = None
        self.color_mask = self.color_img.copy()
    
        # 定義橘色的 RGB 範圍
        lower_orange = np.array([0, 10, 180], dtype=np.uint8)
        upper_orange = np.array([100, 200, 255], dtype=np.uint8)
    
        # 建立橘色遮罩
        mask = cv2.inRange(self.color_img, lower_orange, upper_orange)
    
        # 找出橘色像素的座標
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            self.target_exist = False
            self.target_depth = None
            self.target_position = None
            return
    
        # 取得對應像素的深度值（單位：公尺）
        depths = self.depth_img[ys, xs] * 0.001
        valid_mask = (depths > 0) & (depths <= 0.8)
        if np.count_nonzero(valid_mask) == 0:
            self.target_exist = False
            self.target_depth = None
            self.target_position = None
            return
    
        # 有效遮罩處理
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask]
        valid_depths = depths[valid_mask]
    
        valid_mask_img = np.zeros_like(mask)
        valid_mask_img[valid_ys, valid_xs] = 255
        self.color_mask = cv2.bitwise_and(self.color_img, self.color_img, mask=valid_mask_img)
        alpha = 0.7
        self.color_img = cv2.addWeighted(self.color_img, alpha, self.color_mask, 1 - alpha, 0)
    
        # 計算平均像素位置與深度
        mean_x = np.mean(valid_xs)
        mean_y = np.mean(valid_ys)
        mean_depth = np.mean(valid_depths)
    
        norm_x = (mean_x / self.color_img.shape[1]) * 2 - 1
        norm_y = (mean_y / self.color_img.shape[0]) * 2 - 1
    
        self.target_vel = [norm_x - self.target_norm[0], norm_y - self.target_norm[1]]
        self.target_norm = [norm_x, norm_y]
        self.target_depth = mean_depth
    
        half_width = np.tan(np.radians(55) / 2)
        half_height = np.tan(np.radians(55) / 2)
        X, Y, Z = -(norm_y * half_width), (norm_x * half_height), 1.0
        vec = np.array([X, Y, Z])
        vec_normalized = vec / np.linalg.norm(vec)
        self.target_position = vec_normalized * self.target_depth
    
        self.target_exist = True
        self.target_vis_data = (
            mean_x,
            mean_y,
            f"{self.target_depth:.3f} m",
            f"[{self.target_position[0]:.2f} {self.target_position[1]:.2f} {self.target_position[2]:.2f}]"
        )

    def get_hand(self, depth=False, hand="Left"):
        self.hand_vis_data = None
        self.target_vis_data = None
        self.hand_center = None
        img_rgb = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_handedness:
            for hand_idx, handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label
                if label != hand:
                    hand_landmarks = results.multi_hand_landmarks[hand_idx]
                    lmList = []
                    h, w, _ = self.color_img.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append((int(cx), int(cy)))

                    cx, cy = int((lmList[5][0] + lmList[17][0]) / 2), int((lmList[5][1] + lmList[17][1]) / 2)
                    self.hand_center = (cx, cy)
                    norm_x = (cx / w) * 2 - 1
                    norm_y = (cy / h) * 2 - 1
                    self.hand_vel = [norm_x - self.hand_norm[0], norm_y - self.hand_norm[1]]
                    self.hand_norm = [norm_x, norm_y]
                    self.hand_exist = True

                    if depth:
                        self.hand_depth = self.depth_img[int(cy), int(cx)] * 0.001
                        half_width = np.tan(np.radians(55) / 2)
                        half_height = np.tan(np.radians(55) / 2)
                        X, Y, Z = -(norm_y * half_width), (norm_x * half_height), 1.0
                        vec = np.array([X, Y, Z])
                        vec_normalized = vec / np.linalg.norm(vec)
                        self.target_position = vec_normalized * self.hand_depth

                        self.hand_vis_data = (
                            cx,
                            cy,
                            f"{self.hand_depth:.3f} m",
                            f"[{self.target_position[0]:.2f} {self.target_position[1]:.2f} {self.target_position[2]:.2f}]"
                        )
                    return
        else:
            self.hand_exist = False

    def start(self):
        self.pipeline.start(self.config)
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()


# import cv2
# import pyrealsense2 as rs
# import numpy as np
# import mediapipe as mp
# class Camera():
#     def __init__(self):
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 60Hz
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30) # 60Hz
#         self.pipeline.start(self.config)  # start pipeline with config
#         frames = self.pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         depth_frame = frames.get_depth_frame()
#         self.color_img = cv2.resize(np.asanyarray(color_frame.get_data()), (400, 300), interpolation=cv2.INTER_AREA)
#         self.depth_img = cv2.resize(np.asanyarray(depth_frame.get_data()), (400, 300), interpolation=cv2.INTER_AREA)
#         self.color_mask = self.color_img
#         self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=0.03), cv2.COLORMAP_JET)

#         self.target_exist = False
#         self.target_pixel = [320, 240]
#         self.target_norm = [0.0, 0.0]
#         self.target_vel  = [0.0, 0.0]
#         self.target_depth = 1.0

#         self.colorizer = rs.colorizer()        
#         self.depth_to_disparity = rs.disparity_transform(True)
#         self.disparity_to_depth = rs.disparity_transform(False)

#         self.decimation = rs.decimation_filter()
#         self.spatial = rs.spatial_filter()
#         self.temporal = rs.temporal_filter()
#         self.hole_filling = rs.hole_filling_filter()
#         self.align = rs.align(rs.stream.color)

#         self.pipeline.stop()
#         self.is_running = False

#         self.hand_center = None
#         mp_hands = mp.solutions.hands
#         self.hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.7
#         )
#         self.hand_exist = False
#         self.hand_norm = [0.0, 0.0]
#         self.hand_vel  = [0.0, 0.0]
#         self.hand_depth = 1.0

#         self.target_position = [0.0, 0.0, 0.0]

#     def get_img(self, rgb=True, depth=True):
#         frames = self.pipeline.wait_for_frames()
#         aligned_frames = self.align.process(frames)
#         if rgb == True:
#             color_frame = frames.get_color_frame()
#             # color_frame = aligned_frames.get_color_frame()
#             # self.color_img = np.asanyarray(color_frame.get_data())
#             self.color_img = cv2.resize(np.asanyarray(color_frame.get_data()), (400, 300), interpolation=cv2.INTER_AREA)
#             self.color_img = cv2.rotate(self.color_img, cv2.ROTATE_180)
#         if depth == True:
#             depth_frame = frames.get_depth_frame()
#             depth_frame = aligned_frames.get_depth_frame()

#             depth_frame = self.temporal.process(depth_frame)
#             depth_frame = self.spatial.process(depth_frame)
#             depth_frame = self.hole_filling.process(depth_frame)
#             # self.depth_img = np.asanyarray(depth_frame.get_data())
#             self.depth_img = cv2.resize(np.asanyarray(depth_frame.get_data()), (400, 300), interpolation=cv2.INTER_AREA)
#             self.depth_img = cv2.rotate(self.depth_img, cv2.ROTATE_180)

#             new_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=-0.2), cv2.COLORMAP_JET)
#             self.depth_colormap = cv2.addWeighted(self.depth_colormap, 0.1, new_depth_colormap, 0.9, 0)

#     def show(self, rgb = True, depth = True):
#         if rgb == True:     cv2.imshow("Realsense D435i RGB", cv2.resize(self.color_img, (640, 480), interpolation=cv2.INTER_AREA))
#         if depth == True:   cv2.imshow("Realsense D435i Depth", cv2.resize(self.depth_colormap, (640, 480), interpolation=cv2.INTER_AREA))
#         # if depth == True:   cv2.imshow("Realsense D435i Depth with color", self.depth_img)
#         cv2.waitKey(1)

#     # def get_target(self, depth=False):
#     #     # 定義紅色的RGB範圍
#     #     lower_red = np.array([0, 10, 180], dtype=np.uint8)
#     #     upper_red = np.array([100, 200, 255], dtype=np.uint8)
#     #     # 創建紅色遮罩
#     #     mask = cv2.inRange(self.color_img, lower_red, upper_red)
#     #     self.color_mask = cv2.bitwise_and(self.color_img, self.color_img, mask=mask)
        
#     #     # 將原圖與遮罩層進行混合
#     #     alpha = 0.7  # 透明度
#     #     self.color_img = cv2.addWeighted(self.color_img, alpha, self.color_mask, 1-alpha, 0)

#     #     # 確保在圖像中有紅色物體
#     #     if np.any(mask):
#     #         self.target_exist = True

#     #         # 獲取紅色物體的像素坐標
#     #         coords = np.column_stack(np.where(mask > 0))

#     #         # 計算紅色物體的中心點
#     #         center = np.mean(coords, axis=0)
#     #         center_y, center_x = center

#     #         # 在RGB圖像上畫十字標
#     #         size = 3  # 十字標的大小
#     #         thickness = 1  # 線條的粗細

#     #         # 畫橫線
#     #         cv2.line(self.color_img, (int(center_x) - size, int(center_y)), 
#     #                  (int(center_x) + size, int(center_y)), (255, 255, 255), thickness)
#     #         # 畫縱線
#     #         cv2.line(self.color_img, (int(center_x), int(center_y) - size), 
#     #                  (int(center_x), int(center_y) + size), (255, 255, 255), thickness)

#     #         # 將像素座標轉換至[-1, 1]區間
#     #         norm_x = (center_x / self.color_img.shape[1]) * 2 - 1
#     #         norm_y = (center_y / self.color_img.shape[0]) * 2 - 1

#     #         self.target_vel = [norm_x-self.target_norm[0], norm_y-self.target_norm[1]]
#     #         self.target_norm = [norm_x, norm_y]


#     #         if depth == True:
#     #             # 從深度圖像中獲取對應的深度值
#     #             self.target_depth = self.depth_img[int(center_y), int(center_x)]*0.001  # m

#     #             # 投影平面半寬與半高 FOVx=64, FOVy=50
#     #             half_width = np.tan(np.radians(55) / 2)
#     #             half_height = np.tan(np.radians(55) / 2)
#     #             X, Y, Z = -(norm_y * half_width), (norm_x * half_height), 1.0
#     #             # 單位向量方向
#     #             vec = np.array([X, Y, Z])
#     #             vec_normalized = vec / np.linalg.norm(vec)
#     #             self.target_position = vec_normalized * self.target_depth


#     #             cv2.putText( self.color_img,
#     #                          f"{self.target_depth:.3f} m", 
#     #                          (int(center_x) + 30, int(center_y)), 
#     #                          cv2.FONT_HERSHEY_SIMPLEX, 
#     #                          0.5, (255, 255, 255), 1, cv2.LINE_AA)
#     #             cv2.putText( self.color_img,
#     #                          f"[{self.target_position[0]:.2f} {self.target_position[1]:.2f} {self.target_position[2]:.2f}]", 
#     #                          (int(center_x) + 30, int(center_y) + 15), 
#     #                          cv2.FONT_HERSHEY_SIMPLEX, 
#     #                          0.4, (255, 255, 255), 1, cv2.LINE_AA)

#     #     else:
#     #         self.target_exist = False

#     def get_target(self):
#         # 定義橘色的RGB範圍（可依需要微調）
#         lower_orange = np.array([0, 10, 180], dtype=np.uint8)
#         upper_orange = np.array([100, 200, 255], dtype=np.uint8)

#         # 建立橘色遮罩
#         mask = cv2.inRange(self.color_img, lower_orange, upper_orange)

#         # 找出橘色像素的座標
#         ys, xs = np.where(mask > 0)
#         if len(xs) == 0:
#             self.target_exist = False
#             self.target_depth = None
#             self.target_position = None
#             return

#         # 取得對應像素的深度值，單位：公尺
#         depths = self.depth_img[ys, xs] * 0.001

#         # 建立有效遮罩（只保留 0 < depth ≤ 0.8 公尺的像素）
#         valid_mask = (depths > 0) & (depths <= 0.8)
#         if np.count_nonzero(valid_mask) == 0:
#             self.target_exist = False
#             self.target_depth = None
#             self.target_position = None
#             return

#         # 保留有效像素位置與深度
#         valid_xs = xs[valid_mask]
#         valid_ys = ys[valid_mask]
#         valid_depths = depths[valid_mask]

#         # 遮罩圖（僅包含有效橘色區域）
#         valid_mask_img = np.zeros_like(mask)
#         valid_mask_img[valid_ys, valid_xs] = 255

#         # 建立彩色圖的可視遮罩
#         self.color_mask = cv2.bitwise_and(self.color_img, self.color_img, mask=valid_mask_img)
#         alpha = 0.7
#         self.color_img = cv2.addWeighted(self.color_img, alpha, self.color_mask, 1 - alpha, 0)

#         # 目標存在
#         self.target_exist = True

#         # 計算平均位置與深度
#         mean_x = np.mean(valid_xs)
#         mean_y = np.mean(valid_ys)
#         mean_depth = np.mean(valid_depths)  # 單位：m

#         # 將像素座標轉換至 [-1, 1] 範圍
#         norm_x = (mean_x / self.color_img.shape[1]) * 2 - 1
#         norm_y = (mean_y / self.color_img.shape[0]) * 2 - 1

#         self.target_vel = [norm_x - self.target_norm[0], norm_y - self.target_norm[1]]
#         self.target_norm = [norm_x, norm_y]
#         self.target_depth = mean_depth

#         # 將畫面中心對應至視角方向 (假設 FOVx ≈ FOVy = 55°)
#         half_width = np.tan(np.radians(55) / 2)
#         half_height = np.tan(np.radians(55) / 2)
#         X, Y, Z = -(norm_y * half_width), (norm_x * half_height), 1.0
#         vec = np.array([X, Y, Z])
#         vec_normalized = vec / np.linalg.norm(vec)
#         self.target_position = vec_normalized * self.target_depth

#         # 繪製十字標與文字
#         size = 3
#         thickness = 1
#         cv2.line(self.color_img, (int(mean_x) - size, int(mean_y)), 
#                  (int(mean_x) + size, int(mean_y)), (255, 255, 255), thickness)
#         cv2.line(self.color_img, (int(mean_x), int(mean_y) - size), 
#                  (int(mean_x), int(mean_y) + size), (255, 255, 255), thickness)

#         cv2.putText(self.color_img,
#                     f"{self.target_depth:.3f} m", 
#                     (int(mean_x) + 30, int(mean_y)), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, (255, 255, 255), 1, cv2.LINE_AA)
#         cv2.putText(self.color_img,
#                     f"[{self.target_position[0]:.2f} {self.target_position[1]:.2f} {self.target_position[2]:.2f}]", 
#                     (int(mean_x) + 30, int(mean_y) + 15), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
#     def get_hand(self, depth=False, hand="Left"):
#         color_img = self.color_img
#         self.hand_center = None
#         img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(img_rgb)

#         if results.multi_handedness:
#             for hand_idx, handedness in enumerate(results.multi_handedness):
#                 label = handedness.classification[0].label  # 'Left' or 'Right'
#                 score = handedness.classification[0].score
#                 # if label == 'Left':
#                 if label != hand:
#                     hand_landmarks = results.multi_hand_landmarks[hand_idx]
#                     lmList = []
#                     h, w, _ = color_img.shape
#                     for id, lm in enumerate(hand_landmarks.landmark):
#                         cx, cy = int(lm.x * w), int(lm.y * h)
#                         lmList.append((int(cx), int(cy)))

#                     # 中心點取 5, 17 號關節平均
#                     cx, cy = int((lmList[5][0] + lmList[17][0]) / 2), \
#                              int((lmList[5][1] + lmList[17][1]) / 2)
#                     self.hand_center = (cx, cy)
#                     norm_x = (cx / w) * 2 - 1
#                     norm_y = (cy / h) * 2 - 1
#                     self.hand_vel = [norm_x - self.hand_norm[0], norm_y - self.hand_norm[1]]
#                     self.hand_norm = [norm_x, norm_y]
#                     self.hand_exist = True

#                     # 畫關節點與中心點
#                     cv2.circle(color_img, self.hand_center, 8, (255, 255, 255), -1)
#                     cv2.circle(color_img, lmList[4],  4, (150, 150, 150), -1)
#                     cv2.circle(color_img, lmList[8],  4, (150, 150, 150), -1)
#                     cv2.circle(color_img, lmList[12], 4, (150, 150, 150), -1)
#                     cv2.circle(color_img, lmList[16], 4, (150, 150, 150), -1)
#                     cv2.circle(color_img, lmList[20], 4, (150, 150, 150), -1)
#                     self.color_img = color_img

#                     if depth:
#                         self.hand_depth = self.depth_img[int(cy), int(cx)] * 0.001
    
#                         # 投影平面半寬與半高 FOVx=64, FOVy=50
#                         half_width = np.tan(np.radians(55) / 2)
#                         half_height = np.tan(np.radians(55) / 2)
#                         X, Y, Z = -(norm_y * half_width), (norm_x * half_height), 1.0
#                         # 單位向量方向
#                         vec = np.array([X, Y, Z])
#                         vec_normalized = vec / np.linalg.norm(vec)
#                         self.target_position = vec_normalized * self.hand_depth


#                         cv2.putText( self.color_img,
#                                      f"{self.hand_depth:.3f} m", 
#                                      (int(cx) + 30, int(cy)), 
#                                      cv2.FONT_HERSHEY_SIMPLEX, 
#                                      0.5, (255, 255, 255), 1, cv2.LINE_AA)
#                         cv2.putText( self.color_img,
#                                      f"[{self.target_position[0]:.2f} {self.target_position[1]:.2f} {self.target_position[2]:.2f}]", 
#                                      (int(cx) + 30, int(cy) + 15), 
#                                      cv2.FONT_HERSHEY_SIMPLEX, 
#                                      0.4, (255, 255, 255), 1, cv2.LINE_AA)
#                     return  # 找到左手就結束

#         else:
#             self.hand_exist = False
        
#     def start(self):
#         self.pipeline.start(self.config)  # start pipeline with config
#         self.is_running = True

#     def stop(self):
#         self.is_running = False
#         self.pipeline.stop()
#         cv2.destroyAllWindows()