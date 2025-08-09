# progress_tracker.py
# Progress tracking and visualization for Speech Practice app

import db
from datetime import datetime, timedelta
from typing import List, Tuple
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGridLayout, QWidget,
    QSizePolicy
)
from PyQt5.QtCore import Qt


class ProgressTrackerDialog(QDialog):
    """Dialog window displaying progress charts for Score, WER, and Clarity metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Progress Tracker")
        self.setModal(False)  # Allow interaction with main window
        self.resize(1000, 600)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Header with controls
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Progress Tracker - View your improvement over time"))
        
        # Time period selector
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
        self.period_combo.setCurrentText("Last 30 days")
        self.period_combo.currentTextChanged.connect(self.update_charts)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_charts)
        
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Time period:"))
        header_layout.addWidget(self.period_combo)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Charts grid
        charts_widget = QWidget()
        charts_layout = QGridLayout(charts_widget)
        
        # Create the three chart widgets
        self.score_chart = self._create_chart("Score", "blue", max_val=5.0)
        self.wer_chart = self._create_chart("Word Error Rate (WER)", "red", max_val=1.0)
        self.clarity_chart = self._create_chart("Clarity", "green", max_val=1.0)
        
        # Add charts to grid (2x2 layout with score taking full width on top)
        charts_layout.addWidget(self.score_chart, 0, 0, 1, 2)  # Score spans both columns
        charts_layout.addWidget(self.wer_chart, 1, 0)          # WER bottom left
        charts_layout.addWidget(self.clarity_chart, 1, 1)      # Clarity bottom right
        
        layout.addWidget(charts_widget)
        
        # Load initial data
        self.update_charts()
    
    def _create_chart(self, title: str, color: str, max_val: float = 1.0) -> pg.PlotWidget:
        """Create a styled chart widget for displaying metric trends."""
        chart = pg.PlotWidget()
        chart.setBackground(pg.mkColor("#151b22"))
        chart.setTitle(title, color="#e6eaf0", size="12pt")
        
        # Configure axes
        chart.setLabel('left', 'Value', color="#e6eaf0")
        chart.setLabel('bottom', 'Date', color="#e6eaf0")
        chart.showGrid(x=True, y=True, alpha=0.3)
        
        # Style the axes
        for axis_name in ['left', 'bottom']:
            axis = chart.getPlotItem().getAxis(axis_name)
            axis.setPen(pg.mkPen("#6f7c91"))
            axis.setTextPen(pg.mkPen("#e6eaf0"))
        
        # Set Y-axis range based on metric type
        if title == "Score":
            chart.setYRange(0, 5.2, padding=0.1)
        elif "WER" in title:
            chart.setYRange(0, 1.05, padding=0.05)
        else:  # Clarity
            chart.setYRange(0, 1.05, padding=0.05)
        
        # Configure plot appearance
        chart.setMouseEnabled(x=True, y=False)
        chart.setMenuEnabled(False)
        
        return chart
    
    def _get_date_range(self, period: str) -> Tuple[datetime, datetime]:
        """Get start and end dates based on selected period."""
        end_date = datetime.now()
        
        if period == "Last 7 days":
            start_date = end_date - timedelta(days=7)
        elif period == "Last 30 days":
            start_date = end_date - timedelta(days=30)
        elif period == "Last 90 days":
            start_date = end_date - timedelta(days=90)
        else:  # All time
            start_date = datetime(2020, 1, 1)  # Far back date
            
        return start_date, end_date
    
    def _get_progress_data(self) -> List[Tuple[datetime, float, float, float]]:
        """Retrieve progress data from database."""
        if not hasattr(self.parent_app, 'db'):
            return []
        
        try:
            sessions = db.get_all_sessions(self.parent_app.db)
            start_date, end_date = self._get_date_range(self.period_combo.currentText())
            
            progress_data = []
            for session in sessions:
                # Parse timestamp
                try:
                    session_date = datetime.strptime(session.timestamp, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    continue
                
                # Filter by date range
                if session_date < start_date or session_date > end_date:
                    continue
                
                # Only include sessions with complete metrics
                if (session.score is not None and 
                    session.wer is not None and 
                    session.clarity is not None):
                    progress_data.append((session_date, session.score, session.wer, session.clarity))
            
            # Sort by date
            progress_data.sort(key=lambda x: x[0])
            return progress_data
            
        except Exception as e:
            print(f"Error retrieving progress data: {e}")
            return []
    
    def update_charts(self):
        """Update all charts with latest data."""
        progress_data = self._get_progress_data()
        
        if not progress_data:
            # Clear charts and show "no data" message
            self._clear_charts()
            return
        
        # Extract data for plotting
        dates = [item[0] for item in progress_data]
        scores = [item[1] for item in progress_data]
        wers = [item[2] for item in progress_data]
        clarities = [item[3] for item in progress_data]
        
        # Convert dates to timestamps for plotting
        timestamps = [date.timestamp() for date in dates]
        
        # Update each chart
        self._update_single_chart(self.score_chart, timestamps, scores, "#00d0ff", dates)
        self._update_single_chart(self.wer_chart, timestamps, wers, "#e5484d", dates)
        self._update_single_chart(self.clarity_chart, timestamps, clarities, "#4ade80", dates)
    
    def _update_single_chart(self, chart: pg.PlotWidget, timestamps: List[float], 
                           values: List[float], color: str, dates: List[datetime]):
        """Update a single chart with data."""
        chart.clear()
        
        if not timestamps or not values:
            return
        
        # Create line plot
        pen = pg.mkPen(color=color, width=2)
        chart.plot(timestamps, values, pen=pen, symbol='o', symbolBrush=color, symbolSize=6)
        
        # Set up time axis
        time_axis = chart.getPlotItem().getAxis('bottom')
        
        # Create time ticks
        if len(dates) > 0:
            # Create reasonable number of ticks
            num_ticks = min(8, len(dates))
            step = max(1, len(dates) // num_ticks)
            
            tick_positions = []
            tick_labels = []
            
            for i in range(0, len(dates), step):
                tick_positions.append(timestamps[i])
                tick_labels.append(dates[i].strftime("%m/%d"))
            
            # Add the last point if it's not already included
            if len(dates) > 1 and step > 1:
                tick_positions.append(timestamps[-1])
                tick_labels.append(dates[-1].strftime("%m/%d"))
            
            ticks = list(zip(tick_positions, tick_labels))
            time_axis.setTicks([ticks])
        
        # Auto-range the x-axis
        if timestamps:
            chart.setXRange(min(timestamps), max(timestamps), padding=0.05)
    
    def _clear_charts(self):
        """Clear all charts when no data is available."""
        for chart in [self.score_chart, self.wer_chart, self.clarity_chart]:
            chart.clear()
            # Add "No data" text
            text_item = pg.TextItem("No data available for selected period", 
                                  anchor=(0.5, 0.5), color="#6f7c91")
            chart.addItem(text_item)
            # Position the text in the center
            chart.autoRange()


def open_progress_tracker(parent_app):
    """Open the progress tracker dialog."""
    dialog = ProgressTrackerDialog(parent_app)
    dialog.show()
    return dialog
