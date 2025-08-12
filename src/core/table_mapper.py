import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Table:
    id: int
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: List[Tuple[int, int]]  # For irregular table shapes
    last_visit_time: Optional[float] = None
    visit_count: int = 0
    current_customers: List[int] = None
    
    def __post_init__(self):
        if self.current_customers is None:
            self.current_customers = []
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        x, y = point
        if self.polygon:
            # Use polygon containment test
            return cv2.pointPolygonTest(np.array(self.polygon), point, False) >= 0
        else:
            # Use bounding box containment
            x1, y1, x2, y2 = self.bbox
            return x1 <= x <= x2 and y1 <= y <= y2
    
    def distance_to_point(self, point: Tuple[int, int]) -> float:
        return np.linalg.norm(np.array(self.center) - np.array(point))


class TableMapper:
    def __init__(self, config_path: Optional[str] = None):
        self.tables: Dict[int, Table] = {}
        self.config_path = config_path
        self.frame_shape = None
        self.calibration_mode = False
        self.temp_points = []
        
        if config_path and Path(config_path).exists():
            self.load_configuration(config_path)
        
        logger.info("TableMapper initialized")
    
    def auto_detect_tables(self, frame: np.ndarray, num_frames: int = 30) -> List[Table]:
        # Simple heuristic-based table detection
        # In practice, this would need more sophisticated detection
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (tables are typically large)
            if area > 5000 and area < 50000:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if shape is roughly rectangular (4-6 vertices)
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    
                    table = Table(
                        id=len(potential_tables),
                        center=center,
                        bbox=(x, y, x + w, y + h),
                        polygon=approx.reshape(-1, 2).tolist()
                    )
                    potential_tables.append(table)
        
        logger.info(f"Auto-detected {len(potential_tables)} potential tables")
        return potential_tables
    
    def manual_calibration(self, frame: np.ndarray) -> np.ndarray:
        self.frame_shape = frame.shape[:2]
        display_frame = frame.copy()
        
        # Draw existing tables
        for table in self.tables.values():
            self._draw_table(display_frame, table)
        
        # Draw temporary calibration points
        for i, point in enumerate(self.temp_points):
            cv2.circle(display_frame, point, 5, (0, 255, 255), -1)
            if i > 0:
                cv2.line(display_frame, self.temp_points[i-1], point, (0, 255, 255), 2)
        
        # Add instructions
        instructions = [
            "Table Calibration Mode",
            "Click to add table corners",
            "Press 'a' to add table",
            "Press 'c' to clear points",
            "Press 'q' to quit calibration"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(display_frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return display_frame
    
    def add_calibration_point(self, x: int, y: int):
        self.temp_points.append((x, y))
        logger.info(f"Added calibration point: ({x}, {y})")
    
    def create_table_from_points(self):
        if len(self.temp_points) >= 3:
            # Create table from calibration points
            points_array = np.array(self.temp_points)
            x, y, w, h = cv2.boundingRect(points_array)
            center = (x + w // 2, y + h // 2)
            
            table_id = len(self.tables)
            table = Table(
                id=table_id,
                center=center,
                bbox=(x, y, x + w, y + h),
                polygon=self.temp_points.copy()
            )
            
            self.tables[table_id] = table
            logger.info(f"Created table {table_id} with {len(self.temp_points)} points")
            
            self.temp_points.clear()
            return table
        else:
            logger.warning("Need at least 3 points to create a table")
            return None
    
    def assign_person_to_table(self, person_position: Tuple[int, int]) -> Optional[int]:
        # Find closest table to person
        min_distance = float('inf')
        assigned_table = None
        
        for table_id, table in self.tables.items():
            if table.contains_point(person_position):
                return table_id
            
            distance = table.distance_to_point(person_position)
            if distance < min_distance:
                min_distance = distance
                assigned_table = table_id
        
        # Only assign if within reasonable distance (e.g., 150 pixels)
        if min_distance < 150:
            return assigned_table
        
        return None
    
    def update_table_visits(self, waiter_position: Tuple[int, int], 
                          timestamp: float, distance_threshold: float = 100.0):
        for table_id, table in self.tables.items():
            distance = table.distance_to_point(waiter_position)
            
            if distance < distance_threshold:
                # Waiter is visiting this table
                if table.last_visit_time is None or (timestamp - table.last_visit_time) > 10.0:
                    # Count as new visit if more than 10 seconds since last visit
                    table.visit_count += 1
                    logger.info(f"Table {table_id} visited (total visits: {table.visit_count})")
                
                table.last_visit_time = timestamp
    
    def get_table_attention_metrics(self, current_time: float) -> Dict:
        metrics = {}
        
        for table_id, table in self.tables.items():
            if table.last_visit_time is not None:
                time_since_visit = current_time - table.last_visit_time
            else:
                time_since_visit = current_time  # Never visited
            
            metrics[table_id] = {
                'visit_count': table.visit_count,
                'time_since_last_visit': time_since_visit,
                'current_customers': len(table.current_customers),
                'needs_attention': time_since_visit > 300  # More than 5 minutes
            }
        
        return metrics
    
    def _draw_table(self, frame: np.ndarray, table: Table, color: Tuple[int, int, int] = (0, 255, 0)):
        if table.polygon:
            points = np.array(table.polygon, np.int32)
            cv2.polylines(frame, [points], True, color, 2)
        else:
            x1, y1, x2, y2 = table.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw table ID
        cv2.putText(frame, f"Table {table.id}", 
                   (table.center[0] - 20, table.center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_tables(self, frame: np.ndarray) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for table_id, table in self.tables.items():
            # Color based on attention needed
            if table.last_visit_time is None:
                color = (0, 0, 255)  # Red - never visited
            else:
                time_since_visit = 0  # Would need current time
                if time_since_visit > 300:
                    color = (0, 165, 255)  # Orange - needs attention
                else:
                    color = (0, 255, 0)  # Green - recently visited
            
            self._draw_table(annotated_frame, table, color)
            
            # Draw visit count
            cv2.putText(annotated_frame, f"Visits: {table.visit_count}",
                       (table.center[0] - 30, table.center[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_frame
    
    def save_configuration(self, path: str):
        config = {
            'tables': []
        }
        
        for table_id, table in self.tables.items():
            table_dict = {
                'id': table.id,
                'center': table.center,
                'bbox': table.bbox,
                'polygon': table.polygon
            }
            config['tables'].append(table_dict)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Table configuration saved to {path}")
    
    def load_configuration(self, path: str):
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.tables.clear()
        
        for table_dict in config.get('tables', []):
            table = Table(
                id=table_dict['id'],
                center=tuple(table_dict['center']),
                bbox=tuple(table_dict['bbox']),
                polygon=table_dict.get('polygon', [])
            )
            self.tables[table.id] = table
        
        logger.info(f"Loaded {len(self.tables)} tables from {path}")
    
    def reset(self):
        for table in self.tables.values():
            table.last_visit_time = None
            table.visit_count = 0
            table.current_customers.clear()
        
        self.temp_points.clear()