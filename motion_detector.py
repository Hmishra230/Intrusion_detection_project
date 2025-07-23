import cv2
import numpy as np
import config
from collections import deque

class TrackedObject:
    def __init__(self, object_id, centroid, bbox):
        self.id = object_id
        self.centroids = deque([centroid], maxlen=30)
        self.bboxes = deque([bbox], maxlen=30)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
        self.last_seen = 0
        self.direction = None
        self.trajectory = []
        self.classification = None

    def update(self, centroid, bbox):
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.kalman.correct(np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]]))
        self.last_seen = 0
        self.trajectory.append(centroid)
        if len(self.centroids) > 1:
            dx = self.centroids[-1][0] - self.centroids[0][0]
            dy = self.centroids[-1][1] - self.centroids[0][1]
            self.direction = np.arctan2(dy, dx)

    def predict(self):
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])

    def get_speed(self):
        if len(self.centroids) < 2:
            return 0
        return np.linalg.norm(np.array(self.centroids[-1]) - np.array(self.centroids[-2]))

    def get_direction(self):
        return self.direction

    def get_trajectory(self):
        return list(self.trajectory)

    def classify_motion(self):
        # Placeholder for ML-based motion classification
        # Example: return 'thrown' or 'natural'
        return 'thrown' if self.get_speed() > config.MIN_SPEED_THRESHOLD else 'natural'

class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.objects = {}
        self.next_id = 1
        self.max_lost = 10
        self.frame_count = 0
        self.adaptive_lr = 0.005  # Adaptive background learning rate

    def background_subtraction(self, frame, roi_mask=None):
        # Adaptive background learning
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.adaptive_lr)
        if roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
        return fg_mask

    def filter_noise(self, fg_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        return opened

    def detect_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def update_tracks(self, detections):
        # Hungarian algorithm or greedy matching for assignment
        assigned = set()
        for det in detections:
            centroid, bbox = det
            min_dist = float('inf')
            min_id = None
            for obj_id, obj in self.objects.items():
                dist = np.linalg.norm(np.array(obj.centroids[-1]) - np.array(centroid))
                if dist < 50 and obj_id not in assigned:
                    if dist < min_dist:
                        min_dist = dist
                        min_id = obj_id
            if min_id is not None:
                self.objects[min_id].update(centroid, bbox)
                assigned.add(min_id)
            else:
                self.objects[self.next_id] = TrackedObject(self.next_id, centroid, bbox)
                self.next_id += 1
        # Mark lost objects
        lost_ids = []
        for obj_id, obj in self.objects.items():
            if obj_id not in assigned:
                obj.last_seen += 1
                obj.kalman.predict()
                if obj.last_seen > self.max_lost:
                    lost_ids.append(obj_id)
        for obj_id in lost_ids:
            del self.objects[obj_id]

    def process(self, frame, roi_mask=None):
        fg_mask = self.background_subtraction(frame, roi_mask)
        filtered_mask = self.filter_noise(fg_mask)
        contours = self.detect_contours(filtered_mask)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = (int(x + w / 2), int(y + h / 2))
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            if area < config.MIN_OBJECT_AREA or area > config.MAX_OBJECT_AREA:
                continue
            if solidity < 0.7:
                continue
            detections.append((centroid, (x, y, w, h)))
        self.update_tracks(detections)
        # Trajectory and direction analysis, ML classification
        results = []
        for obj in self.objects.values():
            speed = obj.get_speed()
            direction = obj.get_direction()
            traj = obj.get_trajectory()
            classification = obj.classify_motion()
            results.append({
                'id': obj.id,
                'centroid': obj.centroids[-1],
                'bbox': obj.bboxes[-1],
                'speed': speed,
                'direction': direction,
                'trajectory': traj,
                'classification': classification
            })
        return results 