import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import time
from .pose_detector_yolo import PoseDetectorYOLO, Keypoints

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




class GestureDetectorGPU:
    """GPU-accelerated gesture detection using YOLOv8 pose estimation"""
    
    def __init__(self, 
                 device: Optional[str] = None,
                 batch_size: int = 8,
                 min_detection_confidence: float = 0.5,
                 hand_raise_threshold: float = 30,  # pixels above shoulder
                 pose_confidence_threshold: float = 0.5):
        
        # GPU configuration
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"GestureDetectorGPU initializing on device: {self.device}")
        
        # Model configuration
        self.batch_size = batch_size
        self.min_detection_confidence = min_detection_confidence
        self.hand_raise_threshold = hand_raise_threshold
        self.pose_confidence_threshold = pose_confidence_threshold
        
        # Initialize YOLOv8 pose detection model
        self.pose_detector = PoseDetectorYOLO(
            model_name='yolov8m-pose.pt',
            device=self.device,
            conf_threshold=pose_confidence_threshold,
            batch_size=batch_size
        )
        
        # Gesture tracking
        self.gesture_events: List[GestureEvent] = []
        self.active_gestures: Dict[int, GestureEvent] = {}
        
        # Detection parameters
        self.hand_raise_duration_threshold = 1.0  # seconds
        self.gesture_cooldown = 5.0  # seconds between gestures from same person
        
        # Performance tracking
        self.processing_times = []
        
        # Pose tracking for temporal stability
        self.person_poses: Dict[int, List[Keypoints]] = {}  # Keep recent poses for smoothing
        self.max_pose_history = 5  # frames
        
        logger.info("✅ GestureDetectorGPU initialized with YOLOv8 pose estimation")
    
    
    
    
    def detect_hand_raise_batch(self, 
                               frame: np.ndarray,
                               persons: Dict[int, any], 
                               frame_num: int, 
                               timestamp: float) -> List[GestureEvent]:
        """Detect hand-raise gestures using YOLOv8 pose estimation"""
        
        if not persons:
            return []
        
        start_time = time.time()
        
        gesture_events = []
        
        # Process each person individually for better accuracy
        for person_id, person in persons.items():
            # Get pose for this specific person
            pose = self.pose_detector.get_pose_for_person(frame, person.bbox)
            
            if pose is None:
                continue
                
            # Store pose history for temporal stability
            if person_id not in self.person_poses:
                self.person_poses[person_id] = []
            
            self.person_poses[person_id].append(pose)
            if len(self.person_poses[person_id]) > self.max_pose_history:
                self.person_poses[person_id].pop(0)
            
            # Analyze pose for hand-raise gesture
            hand_raise_detected, confidence = self._analyze_hand_raise_yolo(pose, person_id)
            
            if hand_raise_detected and confidence > self.min_detection_confidence:
                # Calculate position from pose
                center_x = int((pose.bbox[0] + pose.bbox[2]) // 2)
                center_y = int((pose.bbox[1] + pose.bbox[3]) // 2)
                
                # Check if this is a new gesture or continuation
                gesture_event = self._process_gesture_detection(
                    person_id, center_x, center_y, confidence, frame_num, timestamp
                )
                
                if gesture_event:
                    gesture_events.append(gesture_event)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return gesture_events
    
    def _analyze_hand_raise_yolo(self, pose: Keypoints, person_id: int) -> Tuple[bool, float]:
        """Analyze YOLOv8 pose keypoints for hand-raise gesture"""
        
        # Extract key points with confidence checking
        left_shoulder = pose.left_shoulder    # [x, y, confidence]
        right_shoulder = pose.right_shoulder
        left_wrist = pose.left_wrist
        right_wrist = pose.right_wrist
        left_elbow = pose.left_elbow
        right_elbow = pose.right_elbow
        
        # Check if we have confident keypoint detections
        key_confidences = [
            left_shoulder[2], right_shoulder[2],
            left_wrist[2], right_wrist[2]
        ]
        
        avg_confidence = np.mean(key_confidences)
        if avg_confidence < self.pose_confidence_threshold:
            return False, 0.0
        
        # Calculate shoulder midpoint
        if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
            shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        elif left_shoulder[2] > 0.3:
            shoulder_mid_y = left_shoulder[1]
        elif right_shoulder[2] > 0.3:
            shoulder_mid_y = right_shoulder[1]
        else:
            return False, 0.0
        
        # Check if either wrist is raised significantly above shoulder level
        left_hand_raised = False
        right_hand_raised = False
        max_raise_distance = 0
        
        if left_wrist[2] > 0.4:  # Confident wrist detection
            left_raise_distance = shoulder_mid_y - left_wrist[1]  # Positive = raised
            if left_raise_distance > self.hand_raise_threshold:
                left_hand_raised = True
                max_raise_distance = max(max_raise_distance, left_raise_distance)
        
        if right_wrist[2] > 0.4:  # Confident wrist detection
            right_raise_distance = shoulder_mid_y - right_wrist[1]  # Positive = raised
            if right_raise_distance > self.hand_raise_threshold:
                right_hand_raised = True
                max_raise_distance = max(max_raise_distance, right_raise_distance)
        
        # Additional validation: check elbow position for realistic pose
        valid_pose = True
        if left_hand_raised and left_elbow[2] > 0.3:
            # Left elbow should be between shoulder and wrist for natural pose
            if not (left_wrist[1] < left_elbow[1] < left_shoulder[1]):
                valid_pose = False
        
        if right_hand_raised and right_elbow[2] > 0.3:
            # Right elbow should be between shoulder and wrist for natural pose  
            if not (right_wrist[1] < right_elbow[1] < right_shoulder[1]):
                valid_pose = False
        
        # Use temporal smoothing to avoid false positives
        hand_raised = (left_hand_raised or right_hand_raised) and valid_pose
        
        # Calculate confidence based on raise distance and keypoint confidence
        if hand_raised:
            # Normalize raise distance to [0, 1] range
            distance_confidence = min(max_raise_distance / 100.0, 1.0)  # 100px = 1.0 confidence
            confidence = (distance_confidence + avg_confidence) / 2.0
        else:
            confidence = 0.0
        
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