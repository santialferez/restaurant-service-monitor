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


class PersonTrackerGPU:
    """GPU-optimized version of PersonTracker with batch processing and TensorRT support"""
    
    def __init__(self, 
                 model_size: str = 'yolov8m.pt',
                 conf_threshold: float = 0.6,  # Balanced threshold to avoid over-detection
                 max_age: int = 45,  # Keep tracks longer to reduce ID fragmentation  
                 movement_threshold: float = 4.0,  # Higher threshold for better classification
                 batch_size: int = 8,
                 use_tensorrt: bool = True,
                 use_half_precision: bool = True,
                 nms_threshold: float = 0.3):
        
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
        
        # TensorRT optimization
        if use_tensorrt and torch.cuda.is_available():
            self._optimize_with_tensorrt(model_size)
        
        self.conf_threshold = conf_threshold
        
        # Initialize DeepSORT tracker with optimized parameters
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=2,  # Fewer frames to confirm track (faster detection)
            nms_max_overlap=nms_threshold,  # Configurable NMS threshold
            embedder="mobilenet",  # Use MobileNet embedder
            embedder_gpu=True,  # Enable GPU embedder
            embedder_model_name="mobilenetv2_x1_0",
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Person tracking
        self.persons: Dict[int, Person] = {}
        self.movement_threshold = movement_threshold
        self.frame_buffer = []  # Buffer for batch processing
        
        # Performance tracking
        self.processing_times = {
            'detection': [],
            'tracking': [], 
            'classification': []
        }
        
        # Warm up the model
        self._warmup_model()
        
        logger.info(f"PersonTrackerGPU initialized with {model_size}")
        logger.info(f"Batch size: {batch_size}, Half precision: {self.use_half_precision}")
    
    def _optimize_with_tensorrt(self, model_size: str):
        """Export model to TensorRT for faster inference"""
        try:
            tensorrt_path = model_size.replace('.pt', '_tensorrt.engine')
            
            # Check if TensorRT model already exists
            if not self._tensorrt_model_exists(tensorrt_path):
                logger.info("Exporting YOLOv8 to TensorRT format...")
                
                # Export with optimal settings for RTX 6000
                self.yolo.export(
                    format='engine',
                    half=self.use_half_precision,
                    dynamic=False,  # Static shapes for better optimization
                    workspace=8,    # 8GB workspace for RTX 6000
                    imgsz=640,      # Standard YOLO input size
                    device=0        # First GPU
                )
                
                # Load the optimized model
                self.yolo = YOLO(tensorrt_path)
                logger.info(f"TensorRT optimization complete: {tensorrt_path}")
            else:
                logger.info(f"Loading existing TensorRT model: {tensorrt_path}")
                self.yolo = YOLO(tensorrt_path)
                
        except Exception as e:
            logger.warning(f"TensorRT optimization failed, using PyTorch: {e}")
    
    def _tensorrt_model_exists(self, path: str) -> bool:
        """Check if TensorRT model file exists"""
        import os
        return os.path.exists(path)
    
    def _warmup_model(self):
        """Warm up the model with dummy inputs for optimal performance"""
        logger.info("Warming up GPU model...")
        
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        if self.use_half_precision:
            dummy_input = dummy_input.half()
        
        # Run several warmup iterations
        with torch.no_grad():
            for _ in range(5):
                _ = self.yolo.model(dummy_input)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model warmup complete")
    
    def _preprocess_frames_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess multiple frames for batch inference"""
        batch_tensors = []
        
        for frame in frames:
            # Resize and normalize
            frame_resized = cv2.resize(frame, (640, 640))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.to(self.device)
            
            if self.use_half_precision:
                tensor = tensor.half()
            
            batch_tensors.append(tensor.unsqueeze(0))
        
        # Stack into batch
        batch_tensor = torch.cat(batch_tensors, dim=0)
        return batch_tensor
    
    def detect_persons_batch(self, frames: List[np.ndarray]) -> List[List[Tuple]]:
        """Detect persons in multiple frames using batch processing"""
        if not frames:
            return []
        
        start_time = time.time()
        
        # Use YOLO predict method instead of direct model call to avoid precision issues
        batch_detections = []
        
        for frame in frames:
            frame_detections = []
            
            # Use YOLO predict method which handles precision correctly
            results = self.yolo.predict(frame, verbose=False)
            
            for result in results:
                # Get detections for person class (class 0 in COCO)  
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)  # Ensure float32
                    confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
                    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if cls == 0 and conf > self.conf_threshold:  # Person class
                            # Convert to integer coordinates
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            
                            bbox_xyxy = (x1, y1, x2, y2)
                            bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
                            
                            # Ensure confidence is float32 for DeepSORT compatibility
                            conf_float32 = float(conf)
                            
                            frame_detections.append((bbox_xywh, conf_float32, 'person'))
            
            batch_detections.append(frame_detections)
        
        detection_time = time.time() - start_time
        self.processing_times['detection'].append(detection_time)
        
        return batch_detections
    
    def update_tracks_batch(self, frames: List[np.ndarray], frame_numbers: List[int]) -> List[Dict[int, Person]]:
        """Update tracking for multiple frames"""
        if not frames:
            return []
        
        # Detect persons in batch
        batch_detections = self.detect_persons_batch(frames)
        
        batch_results = []
        
        for frame_idx, (frame, frame_num, detections) in enumerate(zip(frames, frame_numbers, batch_detections)):
            # Update tracking for this frame
            start_time = time.time()
            
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            tracking_time = time.time() - start_time
            self.processing_times['tracking'].append(tracking_time)
            
            # Process tracks into Person objects
            frame_persons = self._process_tracks(tracks, frame_num)
            batch_results.append(frame_persons)
        
        return batch_results
    
    def update_tracks(self, frame: np.ndarray, frame_num: int) -> Dict[int, Person]:
        """Single frame update for compatibility with existing code"""
        batch_results = self.update_tracks_batch([frame], [frame_num])
        return batch_results[0] if batch_results else {}
    
    def _process_tracks(self, tracks, frame_num: int) -> Dict[int, Person]:
        """Process DeepSORT tracks into Person objects"""
        frame_persons = {}
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format
            
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_tuple = (x1, y1, x2, y2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if track_id in self.persons:
                # Update existing person
                person = self.persons[track_id]
                person.update_position(bbox_tuple, frame_num)
                person.confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
            else:
                # Create new person
                person = Person(
                    id=track_id,
                    bbox=bbox_tuple,
                    center=center,
                    confidence=track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5,
                    person_type='unknown',
                    track_history=[center],
                    last_seen_frame=frame_num,
                    first_seen_frame=frame_num
                )
                self.persons[track_id] = person
                logger.info(f"New person detected: ID {track_id}")
            
            frame_persons[track_id] = person
        
        # Classify person types based on movement
        self._classify_person_types()
        
        return frame_persons
    
    def _classify_person_types(self):
        """Classify persons as waiters or customers based on movement patterns"""
        start_time = time.time()
        
        for person_id, person in self.persons.items():
            if len(person.track_history) < 10:  # Need sufficient data
                continue
            
            if person.person_type != 'unknown':  # Already classified
                continue
            
            # Calculate average movement speed
            speeds = []
            for i in range(1, len(person.track_history)):
                prev_pos = np.array(person.track_history[i-1])
                curr_pos = np.array(person.track_history[i])
                speed = np.linalg.norm(curr_pos - prev_pos)
                speeds.append(speed)
            
            if speeds:
                avg_speed = np.mean(speeds)
                
                # Classify based on movement threshold
                if avg_speed > self.movement_threshold:
                    person.person_type = 'waiter'
                    logger.info(f"Person {person_id} classified as waiter (avg movement: {avg_speed:.2f})")
                else:
                    person.person_type = 'customer'
                    logger.info(f"Person {person_id} classified as customer (avg movement: {avg_speed:.2f})")
        
        classification_time = time.time() - start_time
        self.processing_times['classification'].append(classification_time)
    
    def get_waiters(self) -> List[Person]:
        """Get all persons classified as waiters"""
        return [person for person in self.persons.values() if person.person_type == 'waiter']
    
    def get_customers(self) -> List[Person]:
        """Get all persons classified as customers"""
        return [person for person in self.persons.values() if person.person_type == 'customer']
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        
        for operation, times in self.processing_times.items():
            if times:
                stats[operation] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
                }
        
        return stats
    
    def clear_old_tracks(self, current_frame: int, max_age: int = 100):
        """Remove persons that haven't been seen for a while"""
        to_remove = []
        
        for person_id, person in self.persons.items():
            if current_frame - person.last_seen_frame > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.persons[person_id]
            logger.info(f"Removed old person track: ID {person_id}")
    
    def __del__(self):
        """Cleanup GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()