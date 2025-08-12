import cv2
import numpy as np
from typing import Generator, Tuple, Optional, List
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, video_path: str, skip_frames: int = 1, resize_factor: float = 1.0):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.skip_frames = skip_frames
        self.resize_factor = resize_factor
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        
        self._initialize_video()
    
    def _initialize_video(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize_factor)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize_factor)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps:.2f}, Duration: {self.duration:.2f}s")
    
    def process_frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        if not self.cap:
            self._initialize_video()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        
        pbar = tqdm(total=self.total_frames // self.skip_frames, desc="Processing frames")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_count % self.skip_frames == 0:
                if self.resize_factor != 1.0:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                timestamp = frame_count / self.fps
                yield frame_count, frame, timestamp
                pbar.update(1)
            
            frame_count += 1
        
        pbar.close()
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        if not self.cap:
            self._initialize_video()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            if self.resize_factor != 1.0:
                frame = cv2.resize(frame, (self.width, self.height))
            return frame
        return None
    
    def extract_frames_batch(self, start_frame: int, end_frame: int, 
                           batch_size: int = 32) -> List[np.ndarray]:
        frames = []
        for frame_num in range(start_frame, min(end_frame, self.total_frames)):
            frame = self.get_frame(frame_num)
            if frame is not None:
                frames.append(frame)
                if len(frames) >= batch_size:
                    yield frames
                    frames = []
        
        if frames:
            yield frames
    
    def save_frame(self, frame: np.ndarray, output_path: str):
        cv2.imwrite(output_path, frame)
        logger.info(f"Frame saved to: {output_path}")
    
    def create_output_video(self, output_path: str, codec: str = 'mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()
    
    def get_video_info(self) -> dict:
        return {
            'path': str(self.video_path),
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'skip_frames': self.skip_frames
        }