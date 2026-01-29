import numpy as np
import cv2
from collections import deque
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import torch
import torchvision.models as models

class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 1.0
        self.P = np.eye(4) * 10.0
        self.x = np.zeros((4, 1))

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z: np.ndarray):
        z = z.reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()

    def get_state(self) -> np.ndarray:
        return self.x.flatten()

class SimpleFeatureExtractor:
    def __init__(self, feature_dim=128, device='cpu'):
        self.device = device
        self.feature_dim = feature_dim
        try:
            resnet = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.model = torch.nn.Sequential(
                self.model,
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(2048, feature_dim)
            )
        except Exception:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(3 * 64 * 64, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, feature_dim)
            )
        self.model = self.model.to(device)
        self.model.eval()

    def extract(self, bbox_crop: np.ndarray) -> np.ndarray:
        try:
            if bbox_crop is None or bbox_crop.size == 0:
                return np.random.randn(self.feature_dim)
            img = cv2.resize(bbox_crop, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(img_tensor)
            return feature.cpu().numpy().flatten()
        except Exception:
            return np.random.randn(self.feature_dim)

class Track:
    next_id = 1
    def __init__(self, bbox: List[float], feature: np.ndarray = None, feature_dim: int = 128):
        self.id = Track.next_id
        Track.next_id += 1
        self.kf = KalmanFilter()
        cx, cy = self._bbox_center(bbox)
        self.kf.x = np.array([[cx], [cy], [0], [0]])
        self.bbox = bbox
        self.feature_history = deque(maxlen=50)
        if feature is not None:
            self.feature_history.append(feature)
        else:
            self.feature_history.append(np.random.randn(feature_dim))
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.trajectory = deque(maxlen=30)
        self.trajectory.append((cx, cy))
        self.confidence = bbox[4] if len(bbox) > 4 else 1.0
        self.class_id = int(bbox[5]) if len(bbox) > 5 else 0

    def predict(self) -> Tuple[float, float]:
        pos = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return pos

    def update(self, bbox: List[float], feature: np.ndarray = None):
        cx, cy = self._bbox_center(bbox)
        self.kf.update(np.array([cx, cy]))
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.trajectory.append((cx, cy))
        if feature is not None:
            self.feature_history.append(feature)
        self.confidence = bbox[4] if len(bbox) > 4 else 1.0
        self.class_id = int(bbox[5]) if len(bbox) > 5 else 0

    def is_confirmed(self, min_hits=3):
        return self.hits >= min_hits

    def get_feature(self) -> np.ndarray:
        if len(self.feature_history) == 0:
            return None
        return np.mean(list(self.feature_history), axis=0)

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2

    def is_deleted(self, max_age=30):
        return self.time_since_update > max_age

class DeepSORT:
    def __init__(self, max_age: int = 70, min_hits: int = 3, iou_threshold: float = 0.3,
                 feature_dim: int = 128, device: str = 'cpu'):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_dim = feature_dim
        self.device = device
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.feature_extractor = SimpleFeatureExtractor(feature_dim, device)

    def update(self, detections: List[List[float]], frame: np.ndarray) -> List[Dict]:
        self.frame_count += 1
        detection_features = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            try:
                bbox_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                 max(0, x1):min(frame.shape[1], x2)]
                if bbox_crop.size > 0:
                    feature = self.feature_extractor.extract(bbox_crop)
                else:
                    feature = None
            except Exception:
                feature = None
            detection_features.append(feature)
        for track in self.tracks:
            track.predict()
        if len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._match(
                detections, detection_features, self.tracks, frame
            )
            for det_idx, trk_idx in matched:
                self.tracks[trk_idx].update(detections[det_idx], detection_features[det_idx])
            for det_idx in unmatched_dets:
                self.tracks.append(Track(detections[det_idx], detection_features[det_idx], self.feature_dim))
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]
        results = []
        for track in self.tracks:
            if track.is_confirmed(self.min_hits):
                results.append({
                    'id': track.id,
                    'bbox': track.bbox[:4],
                    'confidence': track.confidence,
                    'class_id': track.class_id,
                    'velocity': self._get_velocity(track),
                    'trajectory': list(track.trajectory)
                })
        return results

    def _match(self, detections: List[List[float]], det_features: List[np.ndarray],
               tracks: List[Track], frame: np.ndarray) -> Tuple[List, List, List]:
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        iou_matrix = self._compute_iou_matrix(detections, tracks)
        feature_distances = self._compute_feature_distances(det_features, tracks)
        cost_matrix = 0.7 * (1 - iou_matrix) + 0.3 * feature_distances
        cost_matrix[iou_matrix < self.iou_threshold] = 1e9
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        matched = []
        for d, t in zip(det_indices, trk_indices):
            if cost_matrix[d, t] < 1e9:
                matched.append([d, t])
        unmatched_dets = [d for d in range(len(detections))
                         if d not in set([m[0] for m in matched])]
        unmatched_trks = [t for t in range(len(tracks))
                         if t not in set([m[1] for m in matched])]
        return matched, unmatched_dets, unmatched_trks

    def _compute_iou_matrix(self, detections: List[List[float]],
                           tracks: List[Track]) -> np.ndarray:
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self._iou(det[:4], track.bbox[:4])
        return iou_matrix

    def _compute_feature_distances(self, det_features: List[np.ndarray],
                                  tracks: List[Track]) -> np.ndarray:
        distances = np.ones((len(det_features), len(tracks)))
        for d, det_feat in enumerate(det_features):
            if det_feat is None:
                continue
            for t, track in enumerate(tracks):
                track_feat = track.get_feature()
                if track_feat is not None:
                    dist = 1 - np.dot(det_feat, track_feat) / (
                        np.linalg.norm(det_feat) * np.linalg.norm(track_feat) + 1e-7
                    )
                    distances[d, t] = dist
        return distances

    @staticmethod
    def _iou(bbox1: List[float], bbox2: List[float]) -> float:
        x1_tl, y1_tl, x1_br, y1_br = bbox1
        x2_tl, y2_tl, x2_br, y2_br = bbox2
        x_left = max(x1_tl, x2_tl)
        y_top = max(y1_tl, y2_tl)
        x_right = min(x1_br, x2_br)
        y_bottom = min(y1_br, y2_br)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
        bbox2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
        union = bbox1_area + bbox2_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _get_velocity(track: Track) -> float:
        state = track.kf.get_state()
        vx, vy = state[2], state[3]
        return np.sqrt(vx**2 + vy**2)

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        Track.next_id = 1

class TrackVisualizer:
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    @classmethod
    def draw_tracks(cls, frame: np.ndarray, tracks: List[Dict],
                   show_trajectory: bool = True, show_velocity: bool = True) -> np.ndarray:
        annotated = frame.copy()
        if annotated is None:
            annotated = np.zeros((1, 1, 3), dtype=np.uint8)
        if len(annotated.shape) == 2 or (len(annotated.shape) == 3 and annotated.shape[2] == 1):
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        if annotated.dtype != np.uint8:
            annotated = np.clip(annotated, 0, 255).astype(np.uint8)
        h, w = annotated.shape[:2]
        for track in tracks:
            track_id = track.get('id', 0)
            bbox = track.get('bbox', [0, 0, 0, 0])
            confidence = track.get('confidence', 0.0)
            velocity = track.get('velocity', 0.0)
            trajectory = track.get('trajectory', [])
            color = cls.COLORS[track_id % len(cls.COLORS)]
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id}"
            if show_velocity:
                label += f" V:{velocity:.1f}px/s"
            label += f" {confidence:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx0 = x1
            ty0 = y1 - text_size[1] - 6
            tx1 = x1 + text_size[0] + 4
            ty1 = y1
            tx0 = max(0, min(w - 1, tx0))
            ty0 = max(0, min(h - 1, ty0))
            tx1 = max(0, min(w - 1, tx1))
            ty1 = max(0, min(h - 1, ty1))
            cv2.rectangle(annotated, (tx0, ty0), (tx1, ty1), color, -1)
            text_pos = (tx0 + 2, ty1 - 4)
            cv2.putText(annotated, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if show_trajectory and len(trajectory) > 1:
                points = np.array(trajectory, dtype=np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                cv2.polylines(annotated, [points], False, color, 2)
                if len(trajectory) >= 2:
                    p1 = tuple(map(int, trajectory[-2]))
                    p2 = tuple(map(int, trajectory[-1]))
                    p1 = (max(0, min(w - 1, p1[0])), max(0, min(h - 1, p1[1])))
                    p2 = (max(0, min(w - 1, p2[0])), max(0, min(h - 1, p2[1])))
                    try:
                        cv2.arrowedLine(annotated, p1, p2, color, 2, tipLength=0.3)
                    except Exception:
                        pass
        return annotated
