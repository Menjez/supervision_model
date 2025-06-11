import numpy as np
import cv2

class HomographyHeatmap:
    def __init__(self, frame_width=800, frame_height=520):
        self.model_width = frame_width
        self.model_height = frame_height  #dimensions of pitch space
        self.zone_matrix = np.zeros((self.model_height, self.model_width), dtype=np.uint16) #2d array tht tracks players to build heatmap
        self.homography_matrix = None #transformation matrix that warps points from video space to pitch space
        self.initialized = False
        
    def set_calibration_points(self, video_points, model_points=None): 
        if model_points is None:
            model_points = np.array([
                [0, 0],
                [self.model_width, 0],
                [0, self.model_height],
                [self.model_width, self.model_height]
            ], dtype=np.float32)
            self.homography_matrix, _ = cv2.findHomography(np.array(video_points, dtype=np.float32), model_points)
            self.initialized = True
    
    def add_detections(self, detections_xyxy):
            centroids = detections_xyxy[:, :2] + (detections_xyxy[:, 2:4] - detections_xyxy[:, :2]) / 2
