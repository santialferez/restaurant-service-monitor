import torch
import numpy as np
from ultralytics import YOLO
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Keypoints:
    """Container for human pose keypoints with confidence scores"""
    points: np.ndarray  # Shape: (17, 3) - [x, y, confidence] for each keypoint
    bbox: Tuple[int, int, int, int]  # Person bounding box
    person_id: Optional[int] = None
    
    @property
    def nose(self) -> np.ndarray:
        return self.points[0]
    
    @property
    def left_eye(self) -> np.ndarray:
        return self.points[1]
        
    @property
    def right_eye(self) -> np.ndarray:
        return self.points[2]
        
    @property
    def left_ear(self) -> np.ndarray:
        return self.points[3]
        
    @property
    def right_ear(self) -> np.ndarray:
        return self.points[4]
    
    @property
    def left_shoulder(self) -> np.ndarray:
        return self.points[5]
        
    @property
    def right_shoulder(self) -> np.ndarray:
        return self.points[6]
        
    @property
    def left_elbow(self) -> np.ndarray:
        return self.points[7]
        
    @property
    def right_elbow(self) -> np.ndarray:
        return self.points[8]
        
    @property
    def left_wrist(self) -> np.ndarray:
        return self.points[9]
        
    @property
    def right_wrist(self) -> np.ndarray:
        return self.points[10]
        
    @property
    def left_hip(self) -> np.ndarray:
        return self.points[11]
        
    @property
    def right_hip(self) -> np.ndarray:
        return self.points[12]
        
    @property
    def left_knee(self) -> np.ndarray:
        return self.points[13]
        
    @property
    def right_knee(self) -> np.ndarray:
        return self.points[14]
        
    @property
    def left_ankle(self) -> np.ndarray:
        return self.points[15]
        
    @property
    def right_ankle(self) -> np.ndarray:
        return self.points[16]


class PoseDetectorYOLO:
    """YOLOv8-based pose detection with GPU acceleration"""
    
    def __init__(self, 
                 model_name: str = 'yolov8m-pose.pt',
                 device: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 batch_size: int = 8):
        
        # GPU configuration
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"PoseDetectorYOLO initializing on device: {self.device}")
        
        # Model configuration
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        
        # Load YOLOv8 pose model
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Enable half precision for GPU optimization
        if self.device == 'cuda':
            try:
                self.model.half()
                logger.info("✅ FP16 half precision enabled")
            except Exception as e:
                logger.warning(f"Could not enable FP16: {e}")
        
        # COCO pose keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Performance tracking
        self.processing_times = []
        
        logger.info(f"✅ PoseDetectorYOLO loaded: {model_name}")
        
    def detect_poses(self, frame: np.ndarray) -> List[Keypoints]:
        """Detect poses in a single frame"""
        start_time = time.time()
        
        # Run YOLOv8 pose detection
        results = self.model(frame, 
                           conf=self.conf_threshold, 
                           verbose=False, 
                           device=self.device)
        
        poses = []
        
        # Process results
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Extract keypoints and boxes
                keypoints = result.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)
                boxes = result.boxes.xyxy.cpu().numpy()          # Shape: (N, 4)
                confidences = result.boxes.conf.cpu().numpy()    # Shape: (N,)
                
                # Process each detected person
                for i, (kpts, box, conf) in enumerate(zip(keypoints, boxes, confidences)):
                    # Only keep high-confidence detections
                    if conf < self.conf_threshold:
                        continue
                    
                    # Convert box to integer coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Create Keypoints object
                    pose = Keypoints(
                        points=kpts,  # Shape: (17, 3)
                        bbox=(x1, y1, x2, y2),
                        person_id=None  # Will be assigned by tracker
                    )
                    
                    poses.append(pose)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return poses
    
    def detect_poses_batch(self, frames: List[np.ndarray]) -> List[List[Keypoints]]:
        """Detect poses in multiple frames using batch processing"""
        if not frames:
            return []
        
        start_time = time.time()
        
        # Run batch inference
        results = self.model(frames, 
                           conf=self.conf_threshold, 
                           verbose=False,
                           device=self.device)
        
        all_poses = []
        
        # Process results for each frame
        for frame_idx, result in enumerate(results):
            frame_poses = []
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Extract keypoints and boxes
                keypoints = result.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)
                boxes = result.boxes.xyxy.cpu().numpy()          # Shape: (N, 4)  
                confidences = result.boxes.conf.cpu().numpy()    # Shape: (N,)
                
                # Process each detected person in this frame
                for kpts, box, conf in zip(keypoints, boxes, confidences):
                    if conf < self.conf_threshold:
                        continue
                    
                    # Convert box to integer coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Create Keypoints object
                    pose = Keypoints(
                        points=kpts,  # Shape: (17, 3)
                        bbox=(x1, y1, x2, y2),
                        person_id=None
                    )
                    
                    frame_poses.append(pose)
            
            all_poses.append(frame_poses)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return all_poses
    
    def get_pose_for_person(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Optional[Keypoints]:
        """Extract pose for a specific person given their bounding box"""
        x1, y1, x2, y2 = person_bbox
        
        # Expand ROI slightly for better pose detection
        margin = 20
        roi_x1 = max(0, x1 - margin)
        roi_y1 = max(0, y1 - margin) 
        roi_x2 = min(frame.shape[1], x2 + margin)
        roi_y2 = min(frame.shape[0], y2 + margin)
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return None
        
        # Detect poses in ROI
        roi_poses = self.detect_poses(roi)
        
        if not roi_poses:
            return None
        
        # Get the best pose (highest confidence bbox)
        best_pose = max(roi_poses, key=lambda p: np.mean(p.points[:, 2]))  # Highest avg keypoint confidence
        
        # Adjust keypoint coordinates back to full frame
        adjusted_points = best_pose.points.copy()
        adjusted_points[:, 0] += roi_x1  # Adjust X coordinates
        adjusted_points[:, 1] += roi_y1  # Adjust Y coordinates
        
        # Adjust bbox coordinates
        pose_x1, pose_y1, pose_x2, pose_y2 = best_pose.bbox
        adjusted_bbox = (
            pose_x1 + roi_x1,
            pose_y1 + roi_y1, 
            pose_x2 + roi_x1,
            pose_y2 + roi_y1
        )
        
        return Keypoints(
            points=adjusted_points,
            bbox=adjusted_bbox,
            person_id=None
        )
    
    def visualize_pose(self, frame: np.ndarray, keypoints: Keypoints, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw pose keypoints and skeleton on frame"""
        # Define skeleton connections (COCO format)
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # torso
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],         # arms and face
            [2, 4], [3, 5], [4, 6], [5, 7]                    # head connections
        ]
        
        # Convert to 0-indexed 
        skeleton = [[p1-1, p2-1] for p1, p2 in skeleton]
        
        points = keypoints.points
        
        # Draw skeleton connections
        for p1_idx, p2_idx in skeleton:
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            
            # Only draw if both points are confident
            if p1[2] > 0.3 and p2[2] > 0.3:
                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints  
        for i, point in enumerate(points):
            if point[2] > 0.3:  # Only draw confident keypoints
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 4, color, -1)
                # cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return frame
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
            
        times = np.array(self.processing_times)
        return {
            'avg_processing_time': np.mean(times),
            'min_processing_time': np.min(times),
            'max_processing_time': np.max(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
            'total_detections': len(self.processing_times)
        }
    
    def __del__(self):
        """Clean up GPU memory"""
        if hasattr(self, 'model') and torch.cuda.is_available():
            torch.cuda.empty_cache()