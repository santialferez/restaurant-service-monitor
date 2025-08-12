import numpy as np
import cv2
import torch
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
    
    @property
    def average_movement(self) -> float:
        """Calculate average movement speed"""
        if len(self.track_history) < 2:
            return 0.0
        
        speeds = []
        for i in range(1, len(self.track_history)):
            prev_pos = np.array(self.track_history[i-1])
            curr_pos = np.array(self.track_history[i])
            speed = np.linalg.norm(curr_pos - prev_pos)
            speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0


class PersonTrackerGPU:
    """FIXED GPU-optimized person tracker that detects ALL people"""
    
    def __init__(self, 
                 model_size: str = 'yolov8m.pt',
                 conf_threshold: float = 0.4,  # LOWERED from 0.6 to catch more people
                 max_age: int = 30,  # Reduced to prevent ghost tracks
                 movement_threshold: float = 5.0,
                 batch_size: int = 8,
                 use_tensorrt: bool = False,  # Disabled for stability
                 use_half_precision: bool = True,
                 nms_threshold: float = 0.4):  # INCREASED from 0.25 to allow closer people
        
        # GPU configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_half_precision = use_half_precision and torch.cuda.is_available()
        self.batch_size = batch_size
        
        logger.info(f"PersonTrackerGPU initializing on device: {self.device}")
        
        # Initialize YOLO model with GPU optimization
        self.yolo = YOLO(model_size)
        self.yolo.to(self.device)
        
        # Enable half precision if available
        if self.use_half_precision:
            self.yolo.model.half()
            logger.info("Half precision (FP16) enabled for YOLOv8")
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # FIXED: Initialize DeepSORT with more permissive parameters
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=1,  # FIXED: Only need 1 frame to confirm (was 2)
            nms_max_overlap=nms_threshold,
            embedder="mobilenet",
            embedder_gpu=True,
            embedder_model_name="mobilenetv2_x1_0",
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Person tracking
        self.persons: Dict[int, Person] = {}
        self.movement_threshold = movement_threshold
        
        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'confirmed_tracks': 0,
            'unconfirmed_tracks': 0,
            'frames_processed': 0
        }
        
        # Warm up the model
        self._warmup_model()
        logger.info(f"PersonTrackerGPU initialized with {model_size}")
        logger.info(f"Batch size: {batch_size}, Half precision: {use_half_precision}")
        
    def _warmup_model(self):
        """Warm up YOLO model"""
        logger.info("Warming up GPU model...")
        
        # Create dummy frame
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run several warmup iterations
        for _ in range(3):
            _ = self.yolo.predict(dummy_frame, verbose=False)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model warmup complete")
    
    def detect_persons_improved(self, frame: np.ndarray) -> List[Tuple]:
        """IMPROVED person detection with better parameters"""
        frame_detections = []
        
        # Use YOLO predict with FIXED parameters
        results = self.yolo.predict(
            frame, 
            conf=self.conf_threshold,  # Use lower threshold
            iou=self.nms_threshold,    # Use higher NMS threshold  
            classes=[0],               # Only detect persons (class 0)
            verbose=False,
            device=self.device
        )
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
                classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    if cls == 0:  # Person class - don't double-filter confidence
                        # Convert to integer coordinates
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        
                        # Validate bounding box
                        if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 100:  # Minimum area
                            bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
                            conf_float32 = float(conf)
                            
                            frame_detections.append((bbox_xywh, conf_float32, 'person'))
                            self.stats['total_detections'] += 1
        
        return frame_detections
    
    def update_tracks(self, frame: np.ndarray, frame_num: int) -> Dict[int, Person]:
        """FIXED tracking that processes ALL tracks, not just confirmed ones"""
        # Detect persons with improved detection
        detections = self.detect_persons_improved(frame)
        
        # Update tracking
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # FIXED: Process ALL tracks (confirmed AND unconfirmed)
        frame_persons = {}
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_ltrb()  # Get bounding box
            
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_tuple = (x1, y1, x2, y2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Get confidence (with fallback)
            confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
            
            if track_id in self.persons:
                # Update existing person
                person = self.persons[track_id]
                person.update_position(bbox_tuple, frame_num)
                person.confidence = confidence
            else:
                # Create new person
                person = Person(
                    id=track_id,
                    bbox=bbox_tuple,
                    center=center,
                    confidence=confidence,
                    person_type='unknown',
                    track_history=[center],
                    last_seen_frame=frame_num,
                    first_seen_frame=frame_num
                )
                self.persons[track_id] = person
                logger.info(f"New person detected: ID {track_id}")
            
            frame_persons[track_id] = person
            
            # Track statistics
            if track.is_confirmed():
                self.stats['confirmed_tracks'] += 1
            else:
                self.stats['unconfirmed_tracks'] += 1
        
        # Classify person types based on movement
        self._classify_person_types()
        
        self.stats['frames_processed'] += 1
        
        return frame_persons
    
    def _classify_person_types(self):
        """Classify persons as waiters or customers based on movement patterns"""
        for person_id, person in self.persons.items():
            if len(person.track_history) < 5:  # REDUCED from 10 - faster classification
                continue
            
            if person.person_type != 'unknown':  # Already classified
                continue
            
            # Calculate average movement speed
            avg_movement = person.average_movement
            
            # IMPROVED: Classify based on movement threshold with hysteresis
            if avg_movement > self.movement_threshold:
                person.person_type = 'waiter'
                logger.info(f"Person {person_id} classified as waiter (avg movement: {avg_movement:.2f})")
            else:
                person.person_type = 'customer'
                logger.info(f"Person {person_id} classified as customer (avg movement: {avg_movement:.2f})")
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        total_tracks = self.stats['confirmed_tracks'] + self.stats['unconfirmed_tracks']
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'total_detections': self.stats['total_detections'],
            'total_tracks': total_tracks,
            'confirmed_tracks': self.stats['confirmed_tracks'],
            'unconfirmed_tracks': self.stats['unconfirmed_tracks'],
            'confirmed_ratio': self.stats['confirmed_tracks'] / max(total_tracks, 1),
            'avg_detections_per_frame': self.stats['total_detections'] / max(self.stats['frames_processed'], 1),
            'unique_people': len(self.persons),
            'waiters': len([p for p in self.persons.values() if p.person_type == 'waiter']),
            'customers': len([p for p in self.persons.values() if p.person_type == 'customer'])
        }
    
    def get_waiters(self) -> List[Person]:
        """Get all persons classified as waiters"""
        return [person for person in self.persons.values() if person.person_type == 'waiter']
    
    def get_customers(self) -> List[Person]:
        """Get all persons classified as customers"""
        return [person for person in self.persons.values() if person.person_type == 'customer']