import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

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


class GestureDetector:
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 hand_raise_threshold: float = 0.7):
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
            enable_segmentation=False
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_raise_threshold = hand_raise_threshold
        
        # Gesture events tracking
        self.gesture_events: List[GestureEvent] = []
        self.active_gestures: Dict[int, GestureEvent] = {}
        
        # Hand raise detection parameters
        self.hand_raise_duration_threshold = 1.0  # seconds
        self.gesture_cooldown = 5.0  # seconds between gestures from same person
        self.last_gesture_time: Dict[int, float] = {}
        
        logger.info("GestureDetector initialized with MediaPipe")
    
    def detect_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[object]:
        # Extract person region from frame
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        # Convert BGR to RGB
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_roi)
        
        return results
    
    def is_hand_raised(self, pose_landmarks) -> Tuple[bool, float, str]:
        if not pose_landmarks:
            return False, 0.0, ""
        
        landmarks = pose_landmarks.landmark
        
        # Get key points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check visibility
        min_visibility = 0.5
        
        # Check left hand
        if (left_shoulder.visibility > min_visibility and 
            left_elbow.visibility > min_visibility and 
            left_wrist.visibility > min_visibility):
            
            # Check if left hand is raised (wrist above shoulder)
            if left_wrist.y < left_shoulder.y - 0.1:
                confidence = (left_shoulder.visibility + left_elbow.visibility + left_wrist.visibility) / 3
                return True, confidence, "left_hand"
        
        # Check right hand
        if (right_shoulder.visibility > min_visibility and 
            right_elbow.visibility > min_visibility and 
            right_wrist.visibility > min_visibility):
            
            # Check if right hand is raised (wrist above shoulder)
            if right_wrist.y < right_shoulder.y - 0.1:
                confidence = (right_shoulder.visibility + right_elbow.visibility + right_wrist.visibility) / 3
                return True, confidence, "right_hand"
        
        return False, 0.0, ""
    
    def detect_hand_raise(self, frame: np.ndarray, persons: Dict, 
                         frame_num: int, timestamp: float) -> List[GestureEvent]:
        new_events = []
        
        for person_id, person in persons.items():
            # Skip waiters
            if person.person_type == 'waiter':
                continue
            
            # Check cooldown
            if person_id in self.last_gesture_time:
                if timestamp - self.last_gesture_time[person_id] < self.gesture_cooldown:
                    continue
            
            # Detect pose
            pose_results = self.detect_pose(frame, person.bbox)
            
            if pose_results and pose_results.pose_landmarks:
                is_raised, confidence, hand_type = self.is_hand_raised(pose_results.pose_landmarks)
                
                if is_raised and confidence > self.hand_raise_threshold:
                    # Create new gesture event
                    event = GestureEvent(
                        person_id=person_id,
                        gesture_type=f"hand_raise_{hand_type}",
                        timestamp=timestamp,
                        frame_number=frame_num,
                        position=person.center,
                        confidence=confidence
                    )
                    
                    # Check if this is a continuation of an active gesture
                    if person_id not in self.active_gestures:
                        self.active_gestures[person_id] = event
                        logger.info(f"Hand raise detected: Person {person_id} at frame {frame_num}")
                    else:
                        # Update existing gesture duration
                        self.active_gestures[person_id].confidence = max(
                            self.active_gestures[person_id].confidence, confidence
                        )
                else:
                    # Hand lowered - finalize gesture if it was active
                    if person_id in self.active_gestures:
                        gesture = self.active_gestures[person_id]
                        duration = timestamp - gesture.timestamp
                        
                        if duration >= self.hand_raise_duration_threshold:
                            # Valid gesture - add to events
                            self.gesture_events.append(gesture)
                            new_events.append(gesture)
                            self.last_gesture_time[person_id] = timestamp
                            logger.info(f"Hand raise confirmed: Person {person_id}, duration: {duration:.2f}s")
                        
                        del self.active_gestures[person_id]
        
        return new_events
    
    def check_gesture_response(self, gesture: GestureEvent, waiters: List, 
                              current_timestamp: float, distance_threshold: float = 100.0):
        if gesture.responded:
            return
        
        # Check if any waiter is near the gesture position
        for waiter in waiters:
            distance = np.linalg.norm(
                np.array(waiter.center) - np.array(gesture.position)
            )
            
            if distance < distance_threshold:
                gesture.responded = True
                gesture.response_time = current_timestamp - gesture.timestamp
                logger.info(f"Gesture responded: Person {gesture.person_id}, "
                          f"Response time: {gesture.response_time:.2f}s")
                return
    
    def draw_gestures(self, frame: np.ndarray, persons: Dict) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for person_id, person in persons.items():
            if person_id in self.active_gestures:
                # Draw hand raise indicator
                cv2.circle(annotated_frame, person.center, 20, (0, 255, 255), 3)
                cv2.putText(annotated_frame, "HAND RAISED",
                           (person.center[0] - 50, person.center[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return annotated_frame
    
    def draw_pose_landmarks(self, frame: np.ndarray, persons: Dict) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for person_id, person in persons.items():
            pose_results = self.detect_pose(frame, person.bbox)
            
            if pose_results and pose_results.pose_landmarks:
                # Draw pose landmarks on person ROI
                x1, y1, x2, y2 = person.bbox
                roi_shape = (y2 - y1, x2 - x1)
                
                # Convert normalized coordinates to image coordinates
                for landmark in pose_results.pose_landmarks.landmark:
                    x = int(x1 + landmark.x * (x2 - x1))
                    y = int(y1 + landmark.y * (y2 - y1))
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        
        return annotated_frame
    
    def get_gesture_statistics(self) -> Dict:
        total_gestures = len(self.gesture_events)
        responded_gestures = sum(1 for g in self.gesture_events if g.responded)
        
        response_times = [g.response_time for g in self.gesture_events 
                         if g.responded and g.response_time is not None]
        
        stats = {
            'total_gestures': total_gestures,
            'responded_gestures': responded_gestures,
            'response_rate': responded_gestures / total_gestures if total_gestures > 0 else 0,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'min_response_time': np.min(response_times) if response_times else 0,
            'max_response_time': np.max(response_times) if response_times else 0,
            'median_response_time': np.median(response_times) if response_times else 0
        }
        
        return stats
    
    def finalize_active_gestures(self, current_timestamp: float):
        """Finalize any active gestures at the end of processing"""
        for person_id, gesture in list(self.active_gestures.items()):
            duration = current_timestamp - gesture.timestamp
            
            if duration >= self.hand_raise_duration_threshold:
                # Valid gesture - add to events
                self.gesture_events.append(gesture)
                self.last_gesture_time[person_id] = current_timestamp
                logger.info(f"Hand raise finalized: Person {person_id}, duration: {duration:.2f}s")
            
            del self.active_gestures[person_id]
    
    def reset(self):
        self.gesture_events.clear()
        self.active_gestures.clear()
        self.last_gesture_time.clear()