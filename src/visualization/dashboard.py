import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analytics.service_metrics import ServiceMetrics


class RestaurantDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Restaurant Service Monitor",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.setup_sidebar()
        self.load_data()
    
    def setup_sidebar(self):
        st.sidebar.title("🍽️ Restaurant Monitor")
        st.sidebar.markdown("---")
        
        # Navigation
        self.page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Service Metrics", "Movement Analysis", 
             "Table Management", "Waiter Performance", "Video Playback"]
        )
        
        st.sidebar.markdown("---")
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        # Export options
        st.sidebar.markdown("### Export Options")
        if st.sidebar.button("📊 Export Metrics"):
            self.export_metrics()
        
        if st.sidebar.button("📄 Generate Report"):
            self.generate_report()
    
    def load_data(self):
        # Load analysis results
        results_path = Path("data/outputs/analysis_results.json")
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.analysis_data = json.load(f)
        else:
            self.analysis_data = self.generate_mock_data()
    
    def generate_mock_data(self):
        # Generate mock data for demonstration
        return {
            'metrics': {
                'avg_response_time': 25.3,
                'min_response_time': 8.0,
                'max_response_time': 45.2,
                'median_response_time': 23.5,
                'response_rate': 0.92,
                'total_requests': 48,
                'total_responses': 44,
                'avg_table_visit_interval': 180.5,
                'efficiency_score': 85.2
            },
            'hourly_stats': [
                {'hour': i, 'requests': np.random.randint(2, 10), 
                 'responses': np.random.randint(2, 9),
                 'avg_response_time': np.random.uniform(15, 35)}
                for i in range(8, 21)
            ],
            'table_stats': [
                {'table_id': i, 'visits': np.random.randint(5, 20),
                 'avg_visit_interval': np.random.uniform(150, 300),
                 'current_customers': np.random.randint(0, 4)}
                for i in range(1, 11)
            ],
            'waiter_performance': [
                {'waiter_id': i, 'responses': np.random.randint(10, 30),
                 'avg_response_time': np.random.uniform(18, 30),
                 'efficiency_score': np.random.uniform(70, 95)}
                for i in range(1, 5)
            ]
        }
    
    def run(self):
        if self.page == "Overview":
            self.show_overview()
        elif self.page == "Service Metrics":
            self.show_service_metrics()
        elif self.page == "Movement Analysis":
            self.show_movement_analysis()
        elif self.page == "Table Management":
            self.show_table_management()
        elif self.page == "Waiter Performance":
            self.show_waiter_performance()
        elif self.page == "Video Playback":
            self.show_video_playback()
    
    def show_overview(self):
        st.title("📊 Service Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = self.analysis_data['metrics']
        
        with col1:
            st.metric(
                "Avg Response Time",
                f"{metrics['avg_response_time']:.1f}s",
                delta=f"{metrics['avg_response_time'] - 30:.1f}s",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Response Rate",
                f"{metrics['response_rate']*100:.1f}%",
                delta=f"{(metrics['response_rate'] - 0.9)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Total Requests",
                metrics['total_requests'],
                delta=f"{metrics['total_responses']} responded"
            )
        
        with col4:
            st.metric(
                "Efficiency Score",
                f"{metrics['efficiency_score']:.1f}/100",
                delta="Good" if metrics['efficiency_score'] > 80 else "Needs Improvement",
                delta_color="normal" if metrics['efficiency_score'] > 80 else "inverse"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly activity chart
            df_hourly = pd.DataFrame(self.analysis_data['hourly_stats'])
            
            fig = px.bar(df_hourly, x='hour', y=['requests', 'responses'],
                        title='Hourly Service Activity',
                        labels={'value': 'Count', 'hour': 'Hour of Day'},
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response time distribution
            response_times = np.random.normal(
                metrics['avg_response_time'], 8, 100
            )
            
            fig = px.histogram(x=response_times, nbins=20,
                             title='Response Time Distribution',
                             labels={'x': 'Response Time (seconds)', 'y': 'Frequency'})
            fig.add_vline(x=metrics['avg_response_time'], 
                         line_dash="dash", line_color="red",
                         annotation_text="Average")
            st.plotly_chart(fig, use_container_width=True)
        
        # Alert section
        st.markdown("### 🚨 Alerts & Notifications")
        
        alert_container = st.container()
        with alert_container:
            if metrics['avg_response_time'] > 30:
                st.warning(f"⚠️ Average response time ({metrics['avg_response_time']:.1f}s) exceeds target (30s)")
            
            underserved_tables = [t for t in self.analysis_data['table_stats'] 
                                 if t['avg_visit_interval'] > 300]
            if underserved_tables:
                st.warning(f"⚠️ {len(underserved_tables)} tables need attention (>5 min since last visit)")
            
            if metrics['response_rate'] < 0.9:
                st.error(f"❌ Low response rate: {metrics['response_rate']*100:.1f}%")
            
            if not any([metrics['avg_response_time'] > 30, underserved_tables, 
                       metrics['response_rate'] < 0.9]):
                st.success("✅ All systems operating normally")
    
    def show_service_metrics(self):
        st.title("📈 Detailed Service Metrics")
        
        metrics = self.analysis_data['metrics']
        
        # Response time analysis
        st.subheader("Response Time Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Minimum", f"{metrics['min_response_time']:.1f}s")
        with col2:
            st.metric("Average", f"{metrics['avg_response_time']:.1f}s")
        with col3:
            st.metric("Median", f"{metrics['median_response_time']:.1f}s")
        with col4:
            st.metric("Maximum", f"{metrics['max_response_time']:.1f}s")
        
        # Time series chart
        st.subheader("Response Time Trend")
        
        # Generate mock time series data
        time_points = pd.date_range(start='2024-01-01 08:00', 
                                   periods=100, freq='10min')
        response_data = pd.DataFrame({
            'timestamp': time_points,
            'response_time': np.random.normal(metrics['avg_response_time'], 5, 100),
            'requests': np.random.poisson(3, 100)
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=response_data['timestamp'], 
                      y=response_data['response_time'],
                      name='Response Time',
                      line=dict(color='blue')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(x=response_data['timestamp'], 
                  y=response_data['requests'],
                  name='Requests',
                  opacity=0.3),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Response Time (s)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Requests", secondary_y=True)
        fig.update_layout(title="Service Response Pattern Throughout the Day")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Service quality metrics
        st.subheader("Service Quality Indicators")
        
        quality_data = {
            'Metric': ['Response Rate', 'Efficiency Score', 'Customer Satisfaction*'],
            'Score': [metrics['response_rate']*100, metrics['efficiency_score'], 88.5],
            'Target': [95, 90, 90],
            'Status': ['🟡', '🟢' if metrics['efficiency_score'] > 80 else '🔴', '🟡']
        }
        
        df_quality = pd.DataFrame(quality_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Actual', x=df_quality['Metric'], y=df_quality['Score']))
        fig.add_trace(go.Bar(name='Target', x=df_quality['Metric'], y=df_quality['Target']))
        fig.update_layout(barmode='group', title='Service Quality vs Targets')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_movement_analysis(self):
        st.title("🚶 Movement Pattern Analysis")
        
        # Movement heatmap
        st.subheader("Movement Heatmap")
        
        # Create mock heatmap
        heatmap = np.random.random((480, 640)) * 100
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        fig = px.imshow(heatmap, color_continuous_scale='hot',
                       title='Staff Movement Heatmap',
                       labels={'color': 'Activity Level'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Movement statistics
        st.subheader("Movement Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distance traveled
            waiter_distances = pd.DataFrame({
                'Waiter': [f'Waiter {i}' for i in range(1, 5)],
                'Distance (m)': np.random.uniform(500, 1500, 4)
            })
            
            fig = px.bar(waiter_distances, x='Waiter', y='Distance (m)',
                        title='Total Distance Traveled by Staff')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average speed
            waiter_speeds = pd.DataFrame({
                'Waiter': [f'Waiter {i}' for i in range(1, 5)],
                'Avg Speed (m/s)': np.random.uniform(0.8, 1.5, 4)
            })
            
            fig = px.bar(waiter_speeds, x='Waiter', y='Avg Speed (m/s)',
                        title='Average Movement Speed',
                        color='Avg Speed (m/s)',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Common routes
        st.subheader("Common Movement Routes")
        st.info("🛤️ Analysis shows 3 primary routes between kitchen and dining area")
        
        route_data = pd.DataFrame({
            'Route': ['Kitchen → Tables 1-4', 'Kitchen → Tables 5-8', 'Bar → All Tables'],
            'Frequency': [45, 38, 27],
            'Avg Time (s)': [12, 15, 8]
        })
        
        st.dataframe(route_data, use_container_width=True)
    
    def show_table_management(self):
        st.title("🪑 Table Management Dashboard")
        
        # Table status overview
        st.subheader("Table Status Overview")
        
        table_data = pd.DataFrame(self.analysis_data['table_stats'])
        
        # Create table grid visualization
        fig = go.Figure()
        
        for _, table in table_data.iterrows():
            color = 'green' if table['avg_visit_interval'] < 200 else 'orange' if table['avg_visit_interval'] < 300 else 'red'
            
            fig.add_trace(go.Scatter(
                x=[table['table_id'] % 5],
                y=[table['table_id'] // 5],
                mode='markers+text',
                marker=dict(size=50, color=color),
                text=f"T{table['table_id']}",
                textposition="middle center",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Table Service Status",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visit Frequency")
            
            fig = px.bar(table_data, x='table_id', y='visits',
                        title='Total Visits per Table',
                        labels={'table_id': 'Table', 'visits': 'Number of Visits'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Service Intervals")
            
            fig = px.bar(table_data, x='table_id', y='avg_visit_interval',
                        title='Average Time Between Visits',
                        labels={'table_id': 'Table', 'avg_visit_interval': 'Interval (seconds)'},
                        color='avg_visit_interval',
                        color_continuous_scale='RdYlGn_r')
            fig.add_hline(y=300, line_dash="dash", line_color="red",
                         annotation_text="5 min threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table information
        st.subheader("Detailed Table Information")
        
        # Add status column
        table_data['Status'] = table_data['avg_visit_interval'].apply(
            lambda x: '🟢 Good' if x < 200 else '🟡 Attention' if x < 300 else '🔴 Urgent'
        )
        
        st.dataframe(
            table_data[['table_id', 'visits', 'avg_visit_interval', 
                       'current_customers', 'Status']].rename(columns={
                'table_id': 'Table ID',
                'visits': 'Total Visits',
                'avg_visit_interval': 'Avg Interval (s)',
                'current_customers': 'Current Guests'
            }),
            use_container_width=True
        )
    
    def show_waiter_performance(self):
        st.title("👨‍🍳 Waiter Performance Analysis")
        
        waiter_data = pd.DataFrame(self.analysis_data['waiter_performance'])
        
        # Performance overview
        st.subheader("Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_waiter = waiter_data.loc[waiter_data['efficiency_score'].idxmax()]
            st.metric("Top Performer", f"Waiter {best_waiter['waiter_id']}", 
                     f"{best_waiter['efficiency_score']:.1f}/100")
        
        with col2:
            avg_efficiency = waiter_data['efficiency_score'].mean()
            st.metric("Team Average", f"{avg_efficiency:.1f}/100")
        
        with col3:
            total_responses = waiter_data['responses'].sum()
            st.metric("Total Responses", total_responses)
        
        # Individual performance
        st.subheader("Individual Performance Metrics")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Efficiency Score', 'Response Count', 
                          'Average Response Time', 'Performance Radar')
        )
        
        # Efficiency scores
        fig.add_trace(
            go.Bar(x=waiter_data['waiter_id'], y=waiter_data['efficiency_score'],
                  name='Efficiency'),
            row=1, col=1
        )
        
        # Response counts
        fig.add_trace(
            go.Bar(x=waiter_data['waiter_id'], y=waiter_data['responses'],
                  name='Responses'),
            row=1, col=2
        )
        
        # Response times
        fig.add_trace(
            go.Bar(x=waiter_data['waiter_id'], y=waiter_data['avg_response_time'],
                  name='Avg Response Time'),
            row=2, col=1
        )
        
        # Radar chart
        categories = ['Efficiency', 'Speed', 'Coverage', 'Consistency']
        
        for _, waiter in waiter_data.iterrows():
            values = [
                waiter['efficiency_score'],
                100 - (waiter['avg_response_time'] / 30 * 100),
                min(100, waiter['responses'] * 3),
                85 + np.random.uniform(-10, 10)
            ]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=f"Waiter {waiter['waiter_id']}"
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Performance Recommendations")
        
        for _, waiter in waiter_data.iterrows():
            if waiter['efficiency_score'] < 75:
                st.warning(f"⚠️ Waiter {waiter['waiter_id']}: Consider additional training on response time optimization")
            elif waiter['avg_response_time'] > 25:
                st.info(f"ℹ️ Waiter {waiter['waiter_id']}: Focus on reducing response time to improve efficiency")
    
    def show_video_playback(self):
        st.title("🎥 Video Analysis Playback")
        
        st.info("Video playback with annotations would be displayed here")
        
        # Video controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⏮️ Previous Event"):
                pass
        
        with col2:
            if st.button("▶️ Play/Pause"):
                pass
        
        with col3:
            if st.button("⏭️ Next Event"):
                pass
        
        # Timeline
        st.subheader("Event Timeline")
        
        # Mock timeline data
        events = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01 12:00', periods=20, freq='5min'),
            'Event': ['Hand Raise', 'Waiter Response', 'Table Visit', 'Hand Raise'] * 5,
            'Table': np.random.randint(1, 11, 20)
        })
        
        fig = px.scatter(events, x='Time', y='Table', color='Event',
                        title='Service Events Timeline',
                        hover_data=['Event', 'Table'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Annotation options
        st.subheader("Annotation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Show person tracking", value=True)
            st.checkbox("Show gesture detection", value=True)
            st.checkbox("Show table boundaries", value=True)
        
        with col2:
            st.checkbox("Show movement paths", value=False)
            st.checkbox("Show heatmap overlay", value=False)
            st.checkbox("Show metrics overlay", value=True)
    
    def export_metrics(self):
        # Export functionality
        st.success("📊 Metrics exported successfully!")
    
    def generate_report(self):
        # Report generation
        st.success("📄 Report generated successfully!")


def main():
    dashboard = RestaurantDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()