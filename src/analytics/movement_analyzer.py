import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovementAnalyzer:
    def __init__(self, frame_shape: Tuple[int, int]):
        self.frame_height, self.frame_width = frame_shape
        self.movement_paths: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.heatmap_accumulator = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.speed_data: Dict[int, List[float]] = defaultdict(list)
        self.dwell_times: Dict[Tuple[int, int], float] = defaultdict(float)
        self.grid_size = 50  # For grid-based analysis
        
        logger.info(f"MovementAnalyzer initialized for frame size {frame_shape}")
    
    def update_position(self, person_id: int, position: Tuple[int, int], timestamp: float):
        self.movement_paths[person_id].append(position)
        
        # Update heatmap
        x, y = position
        if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
            # Add gaussian blur for smoother heatmap
            # Convert to int tuple to ensure proper type for OpenCV
            cv2.circle(self.heatmap_accumulator, (int(x), int(y)), 20, 1, -1)
        
        # Calculate speed if we have previous position
        if len(self.movement_paths[person_id]) >= 2:
            prev_pos = self.movement_paths[person_id][-2]
            distance = np.linalg.norm(np.array(position) - np.array(prev_pos))
            # Assuming frame rate for speed calculation (will be adjusted with actual timestamps)
            speed = distance  # pixels per frame
            self.speed_data[person_id].append(speed)
        
        # Update dwell time for grid cell
        grid_x = x // self.grid_size
        grid_y = y // self.grid_size
        self.dwell_times[(grid_x, grid_y)] += 1
    
    def calculate_path_length(self, person_id: int) -> float:
        if person_id not in self.movement_paths or len(self.movement_paths[person_id]) < 2:
            return 0.0
        
        path = self.movement_paths[person_id]
        total_distance = 0.0
        
        for i in range(1, len(path)):
            distance = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            total_distance += distance
        
        return total_distance
    
    def calculate_average_speed(self, person_id: int) -> float:
        if person_id not in self.speed_data or not self.speed_data[person_id]:
            return 0.0
        
        return np.mean(self.speed_data[person_id])
    
    def generate_heatmap(self, normalize: bool = True) -> np.ndarray:
        heatmap = self.heatmap_accumulator.copy()
        
        # Apply gaussian blur for smoothing
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def generate_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        heatmap = self.generate_heatmap()
        
        # Convert heatmap to color
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def get_movement_patterns(self, person_id: int) -> Dict:
        if person_id not in self.movement_paths:
            return {}
        
        path = self.movement_paths[person_id]
        
        if len(path) < 2:
            return {
                'total_distance': 0,
                'avg_speed': 0,
                'coverage_area': 0,
                'predominant_direction': None
            }
        
        # Calculate various metrics
        total_distance = self.calculate_path_length(person_id)
        avg_speed = self.calculate_average_speed(person_id)
        
        # Calculate coverage area (convex hull)
        if len(path) >= 3:
            points = np.array(path, dtype=np.float32)
            try:
                hull = cv2.convexHull(points)
                coverage_area = cv2.contourArea(hull) if hull is not None else 0
            except Exception:
                coverage_area = 0
        else:
            coverage_area = 0
        
        # Calculate predominant movement direction
        directions = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            if dx != 0 or dy != 0:
                angle = np.arctan2(dy, dx)
                directions.append(angle)
        
        if directions:
            # Convert to compass direction
            avg_direction = np.mean(directions)
            compass_direction = self._angle_to_compass(avg_direction)
        else:
            compass_direction = None
        
        return {
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'coverage_area': coverage_area,
            'predominant_direction': compass_direction,
            'path_points': len(path)
        }
    
    def _angle_to_compass(self, angle: float) -> str:
        # Convert angle to compass direction
        angle_deg = np.degrees(angle) % 360
        
        directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        index = int((angle_deg + 22.5) / 45) % 8
        
        return directions[index]
    
    def identify_common_routes(self, min_support: int = 3) -> List[List[Tuple[int, int]]]:
        # Simplified route mining - identify common path segments
        all_segments = []
        
        for path in self.movement_paths.values():
            if len(path) < 10:
                continue
            
            # Create segments of length 5
            for i in range(len(path) - 5):
                segment = path[i:i+5]
                # Discretize to grid
                discretized = [(p[0]//self.grid_size, p[1]//self.grid_size) for p in segment]
                all_segments.append(discretized)
        
        # Find frequent segments
        segment_counts = defaultdict(int)
        for segment in all_segments:
            segment_tuple = tuple(segment)
            segment_counts[segment_tuple] += 1
        
        # Filter by minimum support
        common_routes = []
        for segment, count in segment_counts.items():
            if count >= min_support:
                # Convert back to pixel coordinates
                route = [(p[0] * self.grid_size + self.grid_size//2, 
                         p[1] * self.grid_size + self.grid_size//2) for p in segment]
                common_routes.append(route)
        
        return common_routes
    
    def draw_movement_paths(self, frame: np.ndarray, person_ids: Optional[List[int]] = None,
                           color_map: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
        annotated_frame = frame.copy()
        
        if person_ids is None:
            person_ids = list(self.movement_paths.keys())
        
        for person_id in person_ids:
            if person_id not in self.movement_paths:
                continue
            
            path = self.movement_paths[person_id]
            
            if len(path) < 2:
                continue
            
            # Determine color
            if color_map and person_id in color_map:
                color = color_map[person_id]
            else:
                # Generate random color based on ID
                np.random.seed(person_id)
                color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw path
            points = np.array(path, np.int32)
            cv2.polylines(annotated_frame, [points], False, color, 2)
            
            # Draw start and end points
            if path:
                cv2.circle(annotated_frame, path[0], 5, (0, 255, 0), -1)  # Green start
                cv2.circle(annotated_frame, path[-1], 5, (0, 0, 255), -1)  # Red end
        
        return annotated_frame
    
    def generate_flow_map(self) -> np.ndarray:
        # Create flow visualization showing movement directions
        flow_map = np.zeros((self.frame_height, self.frame_width, 2), dtype=np.float32)
        
        for path in self.movement_paths.values():
            for i in range(1, len(path)):
                prev_pos = path[i-1]
                curr_pos = path[i]
                
                # Calculate flow vector
                flow_x = curr_pos[0] - prev_pos[0]
                flow_y = curr_pos[1] - prev_pos[1]
                
                # Add to flow map
                x, y = curr_pos
                if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                    flow_map[y, x, 0] += flow_x
                    flow_map[y, x, 1] += flow_y
        
        return flow_map
    
    def visualize_flow_field(self, frame: np.ndarray, step: int = 30) -> np.ndarray:
        flow_map = self.generate_flow_map()
        annotated_frame = frame.copy()
        
        # Draw flow vectors
        for y in range(0, self.frame_height, step):
            for x in range(0, self.frame_width, step):
                flow_x = flow_map[y, x, 0]
                flow_y = flow_map[y, x, 1]
                
                if abs(flow_x) > 0.1 or abs(flow_y) > 0.1:
                    # Normalize and scale
                    magnitude = np.sqrt(flow_x**2 + flow_y**2)
                    if magnitude > 0:
                        flow_x = int(flow_x / magnitude * 20)
                        flow_y = int(flow_y / magnitude * 20)
                        
                        # Draw arrow
                        cv2.arrowedLine(annotated_frame, 
                                      (x, y), 
                                      (x + flow_x, y + flow_y),
                                      (255, 255, 0), 2, tipLength=0.3)
        
        return annotated_frame
    
    def get_dwell_time_map(self) -> Dict[Tuple[int, int], float]:
        return dict(self.dwell_times)
    
    def generate_statistics_plot(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Speed distribution
        all_speeds = []
        for speeds in self.speed_data.values():
            all_speeds.extend(speeds)
        
        if all_speeds:
            axes[0, 0].hist(all_speeds, bins=30, edgecolor='black')
            axes[0, 0].set_title('Speed Distribution')
            axes[0, 0].set_xlabel('Speed (pixels/frame)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Path lengths
        path_lengths = [self.calculate_path_length(pid) for pid in self.movement_paths.keys()]
        
        if path_lengths:
            axes[0, 1].bar(range(len(path_lengths)), path_lengths)
            axes[0, 1].set_title('Total Distance Traveled per Person')
            axes[0, 1].set_xlabel('Person ID')
            axes[0, 1].set_ylabel('Distance (pixels)')
        
        # Plot 3: Heatmap
        heatmap = self.generate_heatmap()
        im = axes[1, 0].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[1, 0].set_title('Movement Heatmap')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Dwell time grid
        grid_heatmap = np.zeros((self.frame_height // self.grid_size + 1, 
                                self.frame_width // self.grid_size + 1))
        
        for (gx, gy), count in self.dwell_times.items():
            # Ensure indices are integers
            gx_int = int(gx)
            gy_int = int(gy)
            if 0 <= gx_int < grid_heatmap.shape[1] and 0 <= gy_int < grid_heatmap.shape[0]:
                grid_heatmap[gy_int, gx_int] = count
        
        im = axes[1, 1].imshow(grid_heatmap, cmap='YlOrRd', interpolation='nearest')
        axes[1, 1].set_title('Dwell Time Grid')
        axes[1, 1].set_xlabel('Grid X')
        axes[1, 1].set_ylabel('Grid Y')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Statistics plot saved to {save_path}")
        
        return fig
    
    def export_movement_data(self, filepath: str):
        import pandas as pd
        
        # Prepare data for export
        movement_summary = []
        
        for person_id in self.movement_paths.keys():
            patterns = self.get_movement_patterns(person_id)
            patterns['person_id'] = person_id
            movement_summary.append(patterns)
        
        # Create DataFrame
        df = pd.DataFrame(movement_summary)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Movement data exported to {filepath}")