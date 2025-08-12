import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Person:
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    confidence: float
    person_type: str  # 'waiter' or 'customer'
    track_history: List[Tuple[int, int]]
    last_seen_frame: int
    first_seen_frame: int
    
    def update_position(self, bbox: Tuple[int, int, int, int], frame_num: int):
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.track_history.append(self.center)
        self.last_seen_frame = frame_num
        
        # Limit history to last 100 positions
        if len(self.track_history) > 100:
            self.track_history.pop(0)


class PersonTracker:
    def __init__(self, 
                 model_size: str = 'yolov8m.pt',
                 conf_threshold: float = 0.5,
                 max_age: int = 30,
                 movement_threshold: float = 100.0):
        
        # Initialize YOLO model
        self.yolo = YOLO(model_size)
        self.conf_threshold = conf_threshold
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=max_age, n_init=3, nms_max_overlap=0.5)
        
        # Tracking data
        self.persons: Dict[int, Person] = {}
        self.movement_threshold = movement_threshold
        self.frame_count = 0
        
        # Classification parameters
        self.waiter_movement_history = defaultdict(list)
        self.classification_window = 30  # frames to analyze for classification
        
        logger.info(f"PersonTracker initialized with {model_size}")
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        results = self.yolo(frame, conf=self.conf_threshold, classes=[0])  # class 0 is person
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': 'person'
                    })
        
        return detections
    
    def update_tracks(self, frame: np.ndarray, frame_num: int) -> Dict[int, Person]:
        self.frame_count = frame_num
        
        # Detect persons
        detections = self.detect_persons(frame)
        
        if not detections:
            return self.persons
        
        # Prepare detections for DeepSORT
        # DeepSort expects each detection as: ([x, y, w, h], confidence, class_name)
        raw_detections = []
        for det in detections:
            bbox = det['bbox']
            # Convert to [x, y, width, height] format
            bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            # Add detection in the format DeepSort expects
            raw_detections.append((bbox_xywh, det['confidence'], 'person'))
        
        # Update tracks with detections
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # Update person objects
        current_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            current_ids.add(track_id)
            bbox = track.to_ltrb()  # Get bbox in [left, top, right, bottom] format
            
            if track_id not in self.persons:
                # New person detected
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                self.persons[track_id] = Person(
                    id=track_id,
                    bbox=tuple(map(int, bbox)),
                    center=center,
                    confidence=track.det_conf if hasattr(track, 'det_conf') else 0.5,
                    person_type='unknown',
                    track_history=[center],
                    last_seen_frame=frame_num,
                    first_seen_frame=frame_num
                )
                logger.info(f"New person detected: ID {track_id}")
            else:
                # Update existing person
                self.persons[track_id].update_position(tuple(map(int, bbox)), frame_num)
        
        # Remove lost tracks
        lost_ids = set(self.persons.keys()) - current_ids
        for lost_id in lost_ids:
            if frame_num - self.persons[lost_id].last_seen_frame > 30:
                logger.info(f"Person {lost_id} lost")
                del self.persons[lost_id]
        
        # Classify persons as waiters or customers
        self._classify_persons()
        
        return self.persons
    
    def _classify_persons(self):
        for person_id, person in self.persons.items():
            if person.person_type != 'unknown':
                continue
            
            # Analyze movement patterns
            if len(person.track_history) >= self.classification_window:
                movement_distances = self._calculate_movement_distances(person.track_history)
                avg_movement = np.mean(movement_distances)
                
                # Waiters typically move more than customers
                if avg_movement > self.movement_threshold:
                    person.person_type = 'waiter'
                    logger.info(f"Person {person_id} classified as waiter (avg movement: {avg_movement:.2f})")
                else:
                    person.person_type = 'customer'
                    logger.info(f"Person {person_id} classified as customer (avg movement: {avg_movement:.2f})")
    
    def _calculate_movement_distances(self, positions: List[Tuple[int, int]]) -> List[float]:
        distances = []
        for i in range(1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
            distances.append(dist)
        return distances
    
    def get_waiters(self) -> List[Person]:
        return [p for p in self.persons.values() if p.person_type == 'waiter']
    
    def get_customers(self) -> List[Person]:
        return [p for p in self.persons.values() if p.person_type == 'customer']
    
    def draw_tracks(self, frame: np.ndarray, draw_history: bool = True) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for person_id, person in self.persons.items():
            # Draw bounding box
            color = (0, 255, 0) if person.person_type == 'waiter' else (255, 0, 0)
            cv2.rectangle(annotated_frame, 
                         (person.bbox[0], person.bbox[1]),
                         (person.bbox[2], person.bbox[3]),
                         color, 2)
            
            # Draw ID and type
            label = f"ID:{person_id} ({person.person_type})"
            cv2.putText(annotated_frame, label,
                       (person.bbox[0], person.bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw movement history
            if draw_history and len(person.track_history) > 1:
                points = np.array(person.track_history, dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, color, 2)
        
        return annotated_frame
    
    def get_person_by_position(self, x: int, y: int) -> Optional[Person]:
        for person in self.persons.values():
            if (person.bbox[0] <= x <= person.bbox[2] and 
                person.bbox[1] <= y <= person.bbox[3]):
                return person
        return None
    
    def reset(self):
        self.persons.clear()
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.5)
        self.frame_count = 0
        self.waiter_movement_history.clear()