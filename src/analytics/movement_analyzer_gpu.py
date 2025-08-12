import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovementAnalyzerGPU:
    """GPU-accelerated movement analysis using PyTorch tensors"""
    
    def __init__(self, frame_shape: Tuple[int, int], device: Optional[str] = None):
        self.frame_height, self.frame_width = frame_shape
        
        # GPU configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"MovementAnalyzerGPU initializing on device: {self.device}")
        
        # Movement paths stored as GPU tensors
        self.movement_paths_gpu: Dict[int, torch.Tensor] = {}
        self.movement_paths_cpu: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # GPU-based heatmap accumulator
        self.heatmap_accumulator = torch.zeros(
            (self.frame_height, self.frame_width),
            dtype=torch.float32,
            device=self.device
        )
        
        # Speed and dwell time data
        self.speed_data_gpu: Dict[int, torch.Tensor] = {}
        self.speed_data_cpu: Dict[int, List[float]] = defaultdict(list)
        
        self.dwell_times_gpu = torch.zeros(
            (self.frame_height // 50, self.frame_width // 50),
            dtype=torch.float32,
            device=self.device
        )
        
        self.grid_size = 50  # For grid-based analysis
        
        # Position buffer for batch processing
        self.position_buffer = []
        self.person_id_buffer = []
        self.timestamp_buffer = []
        self.batch_size = 64  # Process positions in batches
        
        # Performance tracking
        self.gpu_memory_usage = []
        
        logger.info(f"MovementAnalyzerGPU initialized for frame size {frame_shape} on {self.device}")
    
    def update_position(self, person_id: int, position: Tuple[int, int], timestamp: float):
        """Update position for a person (batched processing)"""
        self.position_buffer.append(position)
        self.person_id_buffer.append(person_id)
        self.timestamp_buffer.append(timestamp)
        
        # Process batch when buffer is full
        if len(self.position_buffer) >= self.batch_size:
            self._process_position_batch()
        
        # Also maintain CPU version for compatibility
        self.movement_paths_cpu[person_id].append(position)
    
    def _process_position_batch(self):
        """Process a batch of position updates on GPU"""
        if not self.position_buffer:
            return
        
        # Convert to tensors
        positions_tensor = torch.tensor(self.position_buffer, dtype=torch.float32, device=self.device)
        person_ids = np.array(self.person_id_buffer)
        
        # Update heatmap in batch
        self._update_heatmap_batch(positions_tensor)
        
        # Calculate speeds for each person
        self._calculate_speeds_batch(positions_tensor, person_ids)
        
        # Update dwell times in batch
        self._update_dwell_times_batch(positions_tensor)
        
        # Clear buffers
        self.position_buffer.clear()
        self.person_id_buffer.clear()
        self.timestamp_buffer.clear()
        
        # Track GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            self.gpu_memory_usage.append(memory_used)
    
    def _update_heatmap_batch(self, positions: torch.Tensor):
        """Update movement heatmap using GPU operations"""
        # Create gaussian kernels around each position
        x_coords = positions[:, 0].long()
        y_coords = positions[:, 1].long()
        
        # Clamp coordinates to valid range
        x_coords = torch.clamp(x_coords, 0, self.frame_width - 1)
        y_coords = torch.clamp(y_coords, 0, self.frame_height - 1)
        
        # Create gaussian blur effect using tensor operations
        for x, y in zip(x_coords, y_coords):
            # Create a small gaussian kernel around the position
            kernel_size = 20
            sigma = 5.0
            
            # Define kernel bounds
            x_min = max(0, x - kernel_size)
            x_max = min(self.frame_width, x + kernel_size + 1)
            y_min = max(0, y - kernel_size)
            y_max = min(self.frame_height, y + kernel_size + 1)
            
            # Create meshgrid for gaussian
            y_grid, x_grid = torch.meshgrid(
                torch.arange(y_min, y_max, device=self.device),
                torch.arange(x_min, x_max, device=self.device),
                indexing='ij'
            )
            
            # Calculate gaussian values
            gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            
            # Add to heatmap accumulator
            self.heatmap_accumulator[y_min:y_max, x_min:x_max] += gaussian
    
    def _calculate_speeds_batch(self, positions: torch.Tensor, person_ids: np.ndarray):
        """Calculate movement speeds for all persons in batch"""
        unique_persons = np.unique(person_ids)
        
        for person_id in unique_persons:
            # Get positions for this person
            person_mask = person_ids == person_id
            person_positions = positions[person_mask]
            
            if len(person_positions) < 2:
                continue
            
            # Calculate distances between consecutive positions
            pos_diff = person_positions[1:] - person_positions[:-1]
            distances = torch.norm(pos_diff, dim=1)
            
            # Store speeds on GPU
            if person_id not in self.speed_data_gpu:
                self.speed_data_gpu[person_id] = distances
            else:
                self.speed_data_gpu[person_id] = torch.cat([self.speed_data_gpu[person_id], distances])
            
            # Also update CPU version for compatibility
            self.speed_data_cpu[person_id].extend(distances.cpu().numpy().tolist())
    
    def _update_dwell_times_batch(self, positions: torch.Tensor):
        """Update grid-based dwell times using GPU operations"""
        # Convert positions to grid coordinates
        grid_x = (positions[:, 0] / self.grid_size).long()
        grid_y = (positions[:, 1] / self.grid_size).long()
        
        # Clamp to valid grid range
        grid_x = torch.clamp(grid_x, 0, self.dwell_times_gpu.shape[1] - 1)
        grid_y = torch.clamp(grid_y, 0, self.dwell_times_gpu.shape[0] - 1)
        
        # Increment dwell times
        for gx, gy in zip(grid_x, grid_y):
            self.dwell_times_gpu[gy, gx] += 1
    
    def flush_buffers(self):
        """Process any remaining positions in buffer"""
        if self.position_buffer:
            self._process_position_batch()
    
    def calculate_path_length_gpu(self, person_id: int) -> float:
        """Calculate path length using GPU tensors"""
        if person_id not in self.speed_data_gpu:
            return 0.0
        
        speeds = self.speed_data_gpu[person_id]
        if len(speeds) == 0:
            return 0.0
        
        total_distance = torch.sum(speeds).item()
        return total_distance
    
    def calculate_average_speed_gpu(self, person_id: int) -> float:
        """Calculate average speed using GPU tensors"""
        if person_id not in self.speed_data_gpu:
            return 0.0
        
        speeds = self.speed_data_gpu[person_id]
        if len(speeds) == 0:
            return 0.0
        
        avg_speed = torch.mean(speeds).item()
        return avg_speed
    
    def generate_heatmap_gpu(self) -> np.ndarray:
        """Generate movement heatmap using GPU operations"""
        # Normalize heatmap
        heatmap_normalized = self.heatmap_accumulator / torch.max(self.heatmap_accumulator)
        
        # Apply additional gaussian blur for smoother visualization
        # Convert to numpy for cv2 operations
        heatmap_np = heatmap_normalized.cpu().numpy()
        heatmap_blurred = cv2.GaussianBlur(heatmap_np, (15, 15), 5)
        
        return heatmap_blurred
    
    def get_dwell_time_grid_gpu(self) -> np.ndarray:
        """Get dwell time grid as numpy array"""
        return self.dwell_times_gpu.cpu().numpy()
    
    def calculate_movement_patterns_gpu(self) -> Dict[int, Dict]:
        """Calculate comprehensive movement patterns using GPU operations"""
        self.flush_buffers()  # Process any remaining data
        
        patterns = {}
        
        for person_id in self.speed_data_gpu.keys():
            speeds_tensor = self.speed_data_gpu[person_id]
            
            if len(speeds_tensor) == 0:
                continue
            
            # GPU-accelerated statistics
            total_distance = torch.sum(speeds_tensor).item()
            avg_speed = torch.mean(speeds_tensor).item()
            max_speed = torch.max(speeds_tensor).item()
            min_speed = torch.min(speeds_tensor).item()
            speed_std = torch.std(speeds_tensor).item()
            
            # Movement direction analysis (if we have position history)
            direction_analysis = self._analyze_movement_direction_gpu(person_id)
            
            patterns[person_id] = {
                'total_distance': total_distance,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'min_speed': min_speed,
                'speed_std': speed_std,
                'path_points': len(speeds_tensor),
                'direction_analysis': direction_analysis
            }
        
        return patterns
    
    def _analyze_movement_direction_gpu(self, person_id: int) -> Dict:
        """Analyze predominant movement direction using GPU"""
        if person_id not in self.movement_paths_gpu:
            return {'predominant_direction': 'N', 'coverage_area': 0}
        
        positions = self.movement_paths_gpu[person_id]
        
        if len(positions) < 2:
            return {'predominant_direction': 'N', 'coverage_area': 0}
        
        # Calculate direction vectors
        direction_vectors = positions[1:] - positions[:-1]
        
        # Calculate angles
        angles = torch.atan2(direction_vectors[:, 1], direction_vectors[:, 0])
        
        # Convert to degrees
        angles_deg = torch.rad2deg(angles)
        
        # Classify into 8 cardinal directions
        direction_bins = torch.zeros(8, device=self.device)
        
        for angle in angles_deg:
            bin_idx = int((angle + 22.5) % 360 // 45)
            direction_bins[bin_idx] += 1
        
        # Find predominant direction
        predominant_idx = torch.argmax(direction_bins).item()
        directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        predominant_direction = directions[predominant_idx]
        
        # Calculate coverage area
        min_pos = torch.min(positions, dim=0)[0]
        max_pos = torch.max(positions, dim=0)[0]
        coverage_area = torch.prod(max_pos - min_pos).item()
        
        return {
            'predominant_direction': predominant_direction,
            'coverage_area': coverage_area
        }
    
    def generate_flow_field_gpu(self) -> np.ndarray:
        """Generate movement flow field visualization using GPU"""
        self.flush_buffers()
        
        # Create flow field grid
        grid_height = self.frame_height // self.grid_size
        grid_width = self.frame_width // self.grid_size
        
        flow_field = torch.zeros((grid_height, grid_width, 2), device=self.device)
        flow_counts = torch.zeros((grid_height, grid_width), device=self.device)
        
        # Calculate average flow vectors for each grid cell
        for person_id, positions in self.movement_paths_gpu.items():
            if len(positions) < 2:
                continue
            
            # Calculate movement vectors
            movement_vectors = positions[1:] - positions[:-1]
            
            # Map to grid coordinates
            grid_positions = (positions[:-1] / self.grid_size).long()
            
            # Clamp to valid range
            grid_x = torch.clamp(grid_positions[:, 0], 0, grid_width - 1)
            grid_y = torch.clamp(grid_positions[:, 1], 0, grid_height - 1)
            
            # Accumulate flow vectors
            for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
                flow_field[gy, gx] += movement_vectors[i]
                flow_counts[gy, gx] += 1
        
        # Average the flow vectors
        flow_counts[flow_counts == 0] = 1  # Avoid division by zero
        flow_field[:, :, 0] /= flow_counts
        flow_field[:, :, 1] /= flow_counts
        
        return flow_field.cpu().numpy()
    
    def get_gpu_memory_stats(self) -> Dict:
        """Get GPU memory usage statistics"""
        stats = {}
        
        if torch.cuda.is_available() and self.gpu_memory_usage:
            stats = {
                'current_memory_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
                'max_memory_mb': torch.cuda.max_memory_allocated(self.device) / 1024**2,
                'avg_memory_mb': np.mean(self.gpu_memory_usage),
                'peak_memory_mb': max(self.gpu_memory_usage) if self.gpu_memory_usage else 0
            }
        
        return stats
    
    def optimize_memory(self):
        """Optimize GPU memory usage by cleaning up old data"""
        # Limit the length of stored paths to prevent memory overflow
        max_history_length = 1000
        
        for person_id in list(self.speed_data_gpu.keys()):
            if len(self.speed_data_gpu[person_id]) > max_history_length:
                # Keep only recent data
                self.speed_data_gpu[person_id] = self.speed_data_gpu[person_id][-max_history_length:]
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_heatmap_gpu(self, filepath: str):
        """Save heatmap using GPU-generated data"""
        heatmap = self.generate_heatmap_gpu()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Movement Intensity')
        plt.title('Movement Heatmap (GPU Generated)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"GPU-generated heatmap saved to {filepath}")
    
    def __del__(self):
        """Cleanup GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()