import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GestureEvent:
    person_id: int
    gesture_type: str
    timestamp: float
    frame_number: int
    position: Tuple[int, int]
    confidence: float
    table_id: Optional[int] = None
    responded: bool = False
    response_time: Optional[float] = None


class SimplePoseNet(nn.Module):
    """Simplified pose estimation network for hand-raise detection"""
    
    def __init__(self, num_keypoints: int = 17):
        super(SimplePoseNet, self).__init__()
        
        # Use MobileNetV2 as backbone
        self.backbone = mobilenet_v2(pretrained=True).features
        
        # Pose estimation head
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(1280 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_keypoints * 3)  # x, y, confidence for each keypoint
        )
        
    def forward(self, x):
        features = self.backbone(x)
        pose = self.pose_head(features)
        # Reshape to (batch_size, num_keypoints, 3)
        pose = pose.view(x.size(0), -1, 3)
        return pose


class GestureDetectorGPU:
    """GPU-accelerated gesture detection using PyTorch-based pose estimation"""
    
    def __init__(self, 
                 device: Optional[str] = None,
                 batch_size: int = 8,
                 min_detection_confidence: float = 0.5,
                 hand_raise_threshold: float = 0.7):
        
        # GPU configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"GestureDetectorGPU initializing on device: {self.device}")
        
        # Model configuration
        self.batch_size = batch_size
        self.min_detection_confidence = min_detection_confidence
        self.hand_raise_threshold = hand_raise_threshold
        
        # Initialize pose estimation model
        self.pose_model = SimplePoseNet(num_keypoints=17)
        self.pose_model.to(self.device)
        self.pose_model.eval()
        
        # Load pretrained weights if available
        self._load_pretrained_weights()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Gesture tracking
        self.gesture_events: List[GestureEvent] = []
        self.active_gestures: Dict[int, GestureEvent] = {}
        
        # Detection parameters
        self.hand_raise_duration_threshold = 1.0  # seconds
        self.gesture_cooldown = 5.0  # seconds between gestures from same person
        
        # Keypoint indices for COCO format
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Batch processing buffers
        self.roi_buffer = []
        self.person_id_buffer = []
        self.frame_num_buffer = []
        self.timestamp_buffer = []
        
        # Performance tracking
        self.processing_times = []
        
        logger.info("GestureDetectorGPU initialized")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights if available"""
        try:
            # This is a placeholder - in production, you would load actual pretrained weights
            # For now, we'll use the ImageNet pretrained MobileNetV2 backbone
            logger.info("Using ImageNet pretrained MobileNetV2 backbone")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    def preprocess_roi_batch(self, rois: List[np.ndarray]) -> torch.Tensor:
        """Preprocess multiple ROIs for batch processing"""
        batch_tensors = []
        
        for roi in rois:
            if roi is None or roi.size == 0:
                # Create dummy tensor for invalid ROI
                tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            else:
                # Convert BGR to RGB
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                tensor = self.transform(roi_rgb)
            
            batch_tensors.append(tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_tensors, dim=0).to(self.device)
        return batch_tensor
    
    def detect_poses_batch(self, rois: List[np.ndarray]) -> torch.Tensor:
        """Detect poses in multiple ROIs using batch processing"""
        if not rois:
            return torch.empty((0, 17, 3), device=self.device)
        
        # Preprocess ROIs
        batch_tensor = self.preprocess_roi_batch(rois)
        
        # Batch inference
        with torch.no_grad():
            poses = self.pose_model(batch_tensor)
        
        return poses
    
    def detect_hand_raise_batch(self, 
                               frame: np.ndarray,
                               persons: Dict[int, any], 
                               frame_num: int, 
                               timestamp: float) -> List[GestureEvent]:
        """Detect hand-raise gestures using batch processing"""
        
        if not persons:
            return []
        
        start_time = datetime.now()
        
        # Extract ROIs for all persons
        rois = []
        person_ids = []
        person_bboxes = []
        
        for person_id, person in persons.items():
            # Extract ROI from frame
            x1, y1, x2, y2 = person.bbox
            
            # Expand ROI slightly for better pose estimation
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            roi = frame[y1:y2, x1:x2]
            
            rois.append(roi)
            person_ids.append(person_id)
            person_bboxes.append((x1, y1, x2, y2))
        
        # Detect poses in batch
        poses = self.detect_poses_batch(rois)
        
        # Process pose results
        gesture_events = []
        
        for i, (person_id, bbox, pose) in enumerate(zip(person_ids, person_bboxes, poses)):
            # Analyze pose for hand-raise gesture
            hand_raise_detected, confidence = self._analyze_hand_raise_gpu(pose)
            
            if hand_raise_detected and confidence > self.min_detection_confidence:
                # Calculate absolute position
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check if this is a new gesture or continuation
                gesture_event = self._process_gesture_detection(
                    person_id, center_x, center_y, confidence, frame_num, timestamp
                )
                
                if gesture_event:
                    gesture_events.append(gesture_event)
        
        # Track processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_times.append(processing_time)
        
        return gesture_events
    
    def _analyze_hand_raise_gpu(self, pose: torch.Tensor) -> Tuple[bool, float]:
        """Analyze pose keypoints for hand-raise gesture using GPU operations"""
        
        # Extract key points (shoulders, elbows, wrists)
        left_shoulder = pose[5]   # left_shoulder
        right_shoulder = pose[6]  # right_shoulder  
        left_elbow = pose[7]      # left_elbow
        right_elbow = pose[8]     # right_elbow
        left_wrist = pose[9]      # left_wrist
        right_wrist = pose[10]    # right_wrist
        
        # Check confidence for key points
        keypoint_confidences = torch.stack([
            left_shoulder[2], right_shoulder[2], left_elbow[2], 
            right_elbow[2], left_wrist[2], right_wrist[2]
        ])
        
        # Only proceed if we have confident detections
        if torch.mean(keypoint_confidences) < self.min_detection_confidence:
            return False, 0.0
        
        # Calculate shoulder midpoint
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Check if either hand is raised above shoulders
        left_hand_raised = left_wrist[1] < shoulder_mid_y - 0.1  # Y decreases upward
        right_hand_raised = right_wrist[1] < shoulder_mid_y - 0.1
        
        # Calculate confidence based on how high the hand is raised
        left_hand_height = torch.clamp((shoulder_mid_y - left_wrist[1]) * 2, 0, 1)
        right_hand_height = torch.clamp((shoulder_mid_y - right_wrist[1]) * 2, 0, 1)
        
        max_hand_height = torch.max(left_hand_height, right_hand_height)
        
        # Consider additional factors for gesture confidence
        # - Elbow position should be reasonable
        # - Wrist confidence should be high
        
        confidence = max_hand_height.item()
        
        # Apply confidence threshold
        hand_raised = (left_hand_raised or right_hand_raised) and confidence > self.hand_raise_threshold
        
        return hand_raised, confidence
    
    def _process_gesture_detection(self, 
                                 person_id: int, 
                                 center_x: int, 
                                 center_y: int, 
                                 confidence: float,
                                 frame_num: int, 
                                 timestamp: float) -> Optional[GestureEvent]:
        """Process detected gesture and handle temporal tracking"""
        
        if person_id in self.active_gestures:
            # Update existing gesture
            active_gesture = self.active_gestures[person_id]
            active_gesture.confidence = max(active_gesture.confidence, confidence)
            return None
        else:
            # Check cooldown period
            recent_gestures = [g for g in self.gesture_events 
                             if g.person_id == person_id and 
                                timestamp - g.timestamp < self.gesture_cooldown]
            
            if recent_gestures:
                return None  # Still in cooldown
            
            # Create new gesture event
            gesture_event = GestureEvent(
                person_id=person_id,
                gesture_type='hand_raise',
                timestamp=timestamp,
                frame_number=frame_num,
                position=(center_x, center_y),
                confidence=confidence
            )
            
            self.active_gestures[person_id] = gesture_event
            logger.info(f"Hand raise detected: Person {person_id} at frame {frame_num}")
            
            return gesture_event
    
    def finalize_active_gestures(self, current_timestamp: float):
        """Finalize active gestures and add to events list"""
        finalized_gestures = []
        
        for person_id, gesture in list(self.active_gestures.items()):
            gesture_duration = current_timestamp - gesture.timestamp
            
            if gesture_duration >= self.hand_raise_duration_threshold:
                # Gesture is long enough to be considered valid
                self.gesture_events.append(gesture)
                finalized_gestures.append(gesture)
                logger.info(f"Hand raise finalized: Person {person_id}, duration: {gesture_duration:.2f}s")
            
            # Remove from active gestures
            del self.active_gestures[person_id]
        
        return finalized_gestures
    
    def check_gesture_response(self, 
                             gesture: GestureEvent, 
                             waiters: List[any], 
                             current_timestamp: float):
        """Check if gesture has been responded to by waiters"""
        if gesture.responded:
            return
        
        # Simple proximity-based response detection
        for waiter in waiters:
            waiter_pos = waiter.center
            gesture_pos = gesture.position
            
            # Calculate distance
            distance = np.sqrt((waiter_pos[0] - gesture_pos[0])**2 + 
                             (waiter_pos[1] - gesture_pos[1])**2)
            
            # If waiter is close enough (within ~100 pixels), consider it a response
            if distance < 100:
                gesture.responded = True
                gesture.response_time = current_timestamp - gesture.timestamp
                logger.info(f"Gesture responded: Person {gesture.person_id}, "
                          f"Response time: {gesture.response_time:.1f}s")
                break
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'fps': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0,
            'total_gestures': len(self.gesture_events),
            'active_gestures': len(self.active_gestures)
        }
    
    def clear_old_gestures(self, current_timestamp: float, max_age: float = 300.0):
        """Remove old gesture events to prevent memory buildup"""
        initial_count = len(self.gesture_events)
        
        self.gesture_events = [
            gesture for gesture in self.gesture_events
            if current_timestamp - gesture.timestamp < max_age
        ]
        
        removed_count = initial_count - len(self.gesture_events)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old gesture events")
    
    def export_gestures_to_gpu_tensor(self) -> torch.Tensor:
        """Export gesture data as GPU tensor for further analysis"""
        if not self.gesture_events:
            return torch.empty((0, 6), device=self.device)
        
        # Convert gestures to tensor format
        gesture_data = []
        for gesture in self.gesture_events:
            row = [
                gesture.person_id,
                gesture.timestamp,
                gesture.position[0],  # x
                gesture.position[1],  # y
                gesture.confidence,
                1.0 if gesture.responded else 0.0
            ]
            gesture_data.append(row)
        
        return torch.tensor(gesture_data, dtype=torch.float32, device=self.device)
    
    def __del__(self):
        """Cleanup GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()