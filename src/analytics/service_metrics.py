import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServiceEvent:
    timestamp: float
    event_type: str  # 'request', 'response', 'visit'
    table_id: Optional[int]
    person_id: int
    waiter_id: Optional[int] = None
    response_time: Optional[float] = None


@dataclass
class ServiceMetrics:
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    median_response_time: float = 0.0
    response_rate: float = 0.0
    total_requests: int = 0
    total_responses: int = 0
    avg_table_visit_interval: float = 0.0
    busiest_hour: Optional[int] = None
    efficiency_score: float = 0.0


class ServiceMetricsCalculator:
    def __init__(self):
        self.service_events: List[ServiceEvent] = []
        self.response_times: List[float] = []
        self.table_visit_intervals: Dict[int, List[float]] = defaultdict(list)
        self.waiter_performance: Dict[int, Dict] = defaultdict(lambda: {
            'responses': 0,
            'total_response_time': 0.0,
            'tables_served': set()
        })
        
        logger.info("ServiceMetricsCalculator initialized")
    
    def add_request_event(self, timestamp: float, person_id: int, table_id: Optional[int] = None):
        event = ServiceEvent(
            timestamp=timestamp,
            event_type='request',
            table_id=table_id,
            person_id=person_id
        )
        self.service_events.append(event)
        logger.debug(f"Request event added: Person {person_id} at table {table_id}")
    
    def add_response_event(self, request_event: ServiceEvent, waiter_id: int, 
                          response_timestamp: float):
        response_time = response_timestamp - request_event.timestamp
        
        event = ServiceEvent(
            timestamp=response_timestamp,
            event_type='response',
            table_id=request_event.table_id,
            person_id=request_event.person_id,
            waiter_id=waiter_id,
            response_time=response_time
        )
        
        self.service_events.append(event)
        self.response_times.append(response_time)
        
        # Update waiter performance
        self.waiter_performance[waiter_id]['responses'] += 1
        self.waiter_performance[waiter_id]['total_response_time'] += response_time
        if request_event.table_id is not None:
            self.waiter_performance[waiter_id]['tables_served'].add(request_event.table_id)
        
        logger.info(f"Response event: Waiter {waiter_id} responded in {response_time:.2f}s")
    
    def add_table_visit(self, timestamp: float, table_id: int, waiter_id: int):
        event = ServiceEvent(
            timestamp=timestamp,
            event_type='visit',
            table_id=table_id,
            person_id=-1,  # Not specific to a person
            waiter_id=waiter_id
        )
        
        self.service_events.append(event)
        
        # Calculate interval since last visit
        table_visits = [e for e in self.service_events 
                       if e.event_type == 'visit' and e.table_id == table_id]
        
        if len(table_visits) >= 2:
            interval = table_visits[-1].timestamp - table_visits[-2].timestamp
            self.table_visit_intervals[table_id].append(interval)
            logger.debug(f"Table {table_id} visit interval: {interval:.2f}s")
    
    def calculate_metrics(self) -> ServiceMetrics:
        metrics = ServiceMetrics()
        
        # Response time metrics
        if self.response_times:
            metrics.avg_response_time = np.mean(self.response_times)
            metrics.min_response_time = np.min(self.response_times)
            metrics.max_response_time = np.max(self.response_times)
            metrics.median_response_time = np.median(self.response_times)
        
        # Request/Response counts
        requests = [e for e in self.service_events if e.event_type == 'request']
        responses = [e for e in self.service_events if e.event_type == 'response']
        
        metrics.total_requests = len(requests)
        metrics.total_responses = len(responses)
        metrics.response_rate = metrics.total_responses / metrics.total_requests if metrics.total_requests > 0 else 0
        
        # Table visit intervals
        all_intervals = []
        for intervals in self.table_visit_intervals.values():
            all_intervals.extend(intervals)
        
        if all_intervals:
            metrics.avg_table_visit_interval = np.mean(all_intervals)
        
        # Find busiest hour
        if self.service_events:
            hours = [datetime.fromtimestamp(e.timestamp).hour 
                    for e in self.service_events if e.event_type == 'request']
            if hours:
                hour_counts = pd.Series(hours).value_counts()
                metrics.busiest_hour = hour_counts.idxmax()
        
        # Calculate efficiency score (0-100)
        metrics.efficiency_score = self._calculate_efficiency_score(metrics)
        
        return metrics
    
    def _calculate_efficiency_score(self, metrics: ServiceMetrics) -> float:
        score = 100.0
        
        # Penalize for slow response time (target: < 30 seconds)
        if metrics.avg_response_time > 0:
            if metrics.avg_response_time > 30:
                score -= min(30, (metrics.avg_response_time - 30) * 0.5)
        
        # Penalize for low response rate
        score *= metrics.response_rate
        
        # Penalize for long table visit intervals (target: < 5 minutes)
        if metrics.avg_table_visit_interval > 300:
            score -= min(20, (metrics.avg_table_visit_interval - 300) / 30)
        
        return max(0, min(100, score))
    
    def get_waiter_metrics(self) -> Dict[int, Dict]:
        waiter_metrics = {}
        
        for waiter_id, performance in self.waiter_performance.items():
            if performance['responses'] > 0:
                avg_response_time = performance['total_response_time'] / performance['responses']
            else:
                avg_response_time = 0
            
            waiter_metrics[waiter_id] = {
                'total_responses': performance['responses'],
                'avg_response_time': avg_response_time,
                'tables_served': len(performance['tables_served']),
                'efficiency_score': self._calculate_waiter_efficiency(performance, avg_response_time)
            }
        
        return waiter_metrics
    
    def _calculate_waiter_efficiency(self, performance: Dict, avg_response_time: float) -> float:
        score = 100.0
        
        # Factor in response time
        if avg_response_time > 30:
            score -= min(40, (avg_response_time - 30) * 0.8)
        
        # Factor in number of responses (more is better)
        score += min(20, performance['responses'] * 0.5)
        
        # Factor in table coverage
        score += min(10, len(performance['tables_served']) * 2)
        
        return max(0, min(100, score))
    
    def get_hourly_statistics(self) -> pd.DataFrame:
        if not self.service_events:
            return pd.DataFrame()
        
        # Create DataFrame from events
        df = pd.DataFrame([{
            'timestamp': e.timestamp,
            'hour': datetime.fromtimestamp(e.timestamp).hour,
            'event_type': e.event_type,
            'response_time': e.response_time if e.response_time else None
        } for e in self.service_events])
        
        # Group by hour
        hourly_stats = df.groupby('hour').agg({
            'event_type': lambda x: (x == 'request').sum(),
            'response_time': ['mean', 'median', 'count']
        }).round(2)
        
        hourly_stats.columns = ['requests', 'avg_response_time', 'median_response_time', 'responses']
        hourly_stats['response_rate'] = (hourly_stats['responses'] / hourly_stats['requests'] * 100).round(1)
        
        return hourly_stats
    
    def get_table_statistics(self) -> pd.DataFrame:
        table_stats = defaultdict(lambda: {
            'visits': 0,
            'requests': 0,
            'responses': 0,
            'avg_visit_interval': 0,
            'avg_response_time': 0
        })
        
        for event in self.service_events:
            if event.table_id is not None:
                if event.event_type == 'visit':
                    table_stats[event.table_id]['visits'] += 1
                elif event.event_type == 'request':
                    table_stats[event.table_id]['requests'] += 1
                elif event.event_type == 'response':
                    table_stats[event.table_id]['responses'] += 1
        
        # Add interval and response time data
        for table_id, intervals in self.table_visit_intervals.items():
            if intervals:
                table_stats[table_id]['avg_visit_interval'] = np.mean(intervals)
        
        # Calculate average response time per table
        for table_id in table_stats:
            table_responses = [e for e in self.service_events 
                             if e.table_id == table_id and e.event_type == 'response']
            if table_responses:
                response_times = [e.response_time for e in table_responses if e.response_time]
                if response_times:
                    table_stats[table_id]['avg_response_time'] = np.mean(response_times)
        
        return pd.DataFrame.from_dict(table_stats, orient='index')
    
    def generate_summary_report(self) -> str:
        metrics = self.calculate_metrics()
        
        report = f"""
SERVICE METRICS SUMMARY
=======================
Overall Performance:
- Total Requests: {metrics.total_requests}
- Total Responses: {metrics.total_responses}
- Response Rate: {metrics.response_rate:.1%}
- Efficiency Score: {metrics.efficiency_score:.1f}/100

Response Times:
- Average: {metrics.avg_response_time:.1f} seconds
- Minimum: {metrics.min_response_time:.1f} seconds
- Maximum: {metrics.max_response_time:.1f} seconds
- Median: {metrics.median_response_time:.1f} seconds

Table Service:
- Average Visit Interval: {metrics.avg_table_visit_interval:.1f} seconds
- Busiest Hour: {metrics.busiest_hour if metrics.busiest_hour else 'N/A'}
"""
        
        # Add waiter performance
        waiter_metrics = self.get_waiter_metrics()
        if waiter_metrics:
            report += "\nWaiter Performance:\n"
            for waiter_id, stats in waiter_metrics.items():
                report += f"  Waiter {waiter_id}:\n"
                report += f"    - Responses: {stats['total_responses']}\n"
                report += f"    - Avg Response Time: {stats['avg_response_time']:.1f}s\n"
                report += f"    - Tables Served: {stats['tables_served']}\n"
                report += f"    - Efficiency: {stats['efficiency_score']:.1f}/100\n"
        
        return report
    
    def export_metrics(self, filepath: str):
        metrics = self.calculate_metrics()
        waiter_metrics = self.get_waiter_metrics()
        hourly_stats = self.get_hourly_statistics()
        table_stats = self.get_table_statistics()
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Overall metrics
            pd.DataFrame([metrics.__dict__]).to_excel(
                writer, sheet_name='Overall Metrics', index=False
            )
            
            # Waiter metrics
            pd.DataFrame.from_dict(waiter_metrics, orient='index').to_excel(
                writer, sheet_name='Waiter Performance'
            )
            
            # Hourly statistics
            hourly_stats.to_excel(writer, sheet_name='Hourly Statistics')
            
            # Table statistics
            table_stats.to_excel(writer, sheet_name='Table Statistics')
        
        logger.info(f"Metrics exported to {filepath}")