# progress_tracker.py
# Progress tracking and visualization for Speech Practice app

import db
from datetime import datetime
from typing import List, Tuple
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGridLayout, QWidget, QDateEdit
)
from PyQt5.QtCore import Qt, QDate
from error_analytics import (
    generate_feedback_summary,
    get_character_trend_summary,
    get_phoneme_trend_summary,
    get_position_trend_summary,
    get_word_trend_summary,
)


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
        
        # Date range selector
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate.currentDate().addDays(-30))
        self.start_date_edit.dateChanged.connect(self.update_charts)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.dateChanged.connect(self.update_charts)

        # Script filter
        self.script_combo = QComboBox()
        self.script_combo.addItem("All scripts")
        self.script_combo.currentTextChanged.connect(self.update_charts)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_charts)
        
        header_layout.addStretch()
        header_layout.addWidget(QLabel("From:"))
        header_layout.addWidget(self.start_date_edit)
        header_layout.addWidget(QLabel("To:"))
        header_layout.addWidget(self.end_date_edit)
        header_layout.addWidget(QLabel("Script:"))
        header_layout.addWidget(self.script_combo)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)

        # Word-level trend summary block
        trends_group = QtWidgets.QGroupBox("Word Trends")
        trends_layout = QGridLayout(trends_group)
        self.trouble_words_label = QLabel("No trend data yet.")
        self.improved_words_label = QLabel("No trend data yet.")
        self.regressed_words_label = QLabel("No trend data yet.")
        for lbl in [
            self.trouble_words_label,
            self.improved_words_label,
            self.regressed_words_label,
        ]:
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
        trends_layout.addWidget(QLabel("Top Troublesome Words"), 0, 0)
        trends_layout.addWidget(QLabel("Most Improved"), 0, 1)
        trends_layout.addWidget(QLabel("Most Regressed"), 0, 2)
        trends_layout.addWidget(self.trouble_words_label, 1, 0)
        trends_layout.addWidget(self.improved_words_label, 1, 1)
        trends_layout.addWidget(self.regressed_words_label, 1, 2)
        layout.addWidget(trends_group)

        # Character / position / phoneme trend blocks
        lower_trends = QWidget()
        lower_trends_layout = QGridLayout(lower_trends)

        char_group = QtWidgets.QGroupBox("Character Trends")
        char_layout = QVBoxLayout(char_group)
        self.char_kinds_label = QLabel("No character trend data yet.")
        self.char_confusions_label = QLabel("No character confusion data yet.")
        self.char_kinds_label.setWordWrap(True)
        self.char_confusions_label.setWordWrap(True)
        self.char_kinds_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.char_confusions_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        char_layout.addWidget(QLabel("Top Error Kinds"))
        char_layout.addWidget(self.char_kinds_label)
        char_layout.addWidget(QLabel("Top Character Confusions"))
        char_layout.addWidget(self.char_confusions_label)

        pos_group = QtWidgets.QGroupBox("Position Trends")
        pos_layout = QVBoxLayout(pos_group)
        self.position_label = QLabel("No position trend data yet.")
        self.position_improved_label = QLabel("No position delta data yet.")
        self.position_label.setWordWrap(True)
        self.position_improved_label.setWordWrap(True)
        self.position_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.position_improved_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        pos_layout.addWidget(QLabel("Top Error Positions"))
        pos_layout.addWidget(self.position_label)
        pos_layout.addWidget(QLabel("Improvement/Regr."))
        pos_layout.addWidget(self.position_improved_label)

        phon_group = QtWidgets.QGroupBox("Phoneme Trends")
        phon_layout = QVBoxLayout(phon_group)
        self.phoneme_label = QLabel("No phoneme trend data yet.")
        self.phoneme_confusions_label = QLabel("No phoneme confusion data yet.")
        self.phoneme_label.setWordWrap(True)
        self.phoneme_confusions_label.setWordWrap(True)
        self.phoneme_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.phoneme_confusions_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        phon_layout.addWidget(QLabel("Top Trouble Symbols"))
        phon_layout.addWidget(self.phoneme_label)
        phon_layout.addWidget(QLabel("Top Symbol Confusions"))
        phon_layout.addWidget(self.phoneme_confusions_label)

        lower_trends_layout.addWidget(char_group, 0, 0)
        lower_trends_layout.addWidget(pos_group, 0, 1)
        lower_trends_layout.addWidget(phon_group, 0, 2)
        layout.addWidget(lower_trends)

        feedback_group = QtWidgets.QGroupBox("Coaching Focus")
        feedback_layout = QVBoxLayout(feedback_group)
        self.feedback_label = QLabel("No coaching feedback yet.")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        feedback_layout.addWidget(self.feedback_label)
        layout.addWidget(feedback_group)
        
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
    
    def _selected_date_range(self) -> Tuple[datetime, datetime]:
        start_qd = self.start_date_edit.date()
        end_qd = self.end_date_edit.date()
        start_date = datetime(start_qd.year(), start_qd.month(), start_qd.day(), 0, 0, 0)
        end_date = datetime(end_qd.year(), end_qd.month(), end_qd.day(), 23, 59, 59)
        if end_date < start_date:
            start_date, end_date = end_date, start_date
        return start_date, end_date

    def _selected_script(self) -> str | None:
        script_name = self.script_combo.currentText().strip()
        if not script_name or script_name == "All scripts":
            return None
        return script_name

    def _current_filters(self) -> tuple[datetime, datetime, str | None]:
        start_dt, end_dt = self._selected_date_range()
        return start_dt, end_dt, self._selected_script()

    def _refresh_script_options(self) -> None:
        if not hasattr(self.parent_app, "db"):
            return
        try:
            sessions = db.get_all_sessions(self.parent_app.db)
            names = sorted({(s.script_name or "").strip() for s in sessions if (s.script_name or "").strip()})
            current = self.script_combo.currentText().strip() if self.script_combo.count() else "All scripts"
            self.script_combo.blockSignals(True)
            self.script_combo.clear()
            self.script_combo.addItem("All scripts")
            for name in names:
                self.script_combo.addItem(name)
            idx = self.script_combo.findText(current)
            self.script_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.script_combo.blockSignals(False)
        except Exception:
            pass
    
    def _get_progress_data(self) -> List[Tuple[datetime, float, float, float]]:
        """Retrieve progress data from database."""
        if not hasattr(self.parent_app, 'db'):
            return []
        
        try:
            sessions = db.get_all_sessions(self.parent_app.db)
            start_date, end_date = self._selected_date_range()
            selected_script = self._selected_script()
            
            progress_data = []
            for session in sessions:
                # Parse timestamp
                try:
                    session_date = datetime.fromisoformat(session.timestamp)
                except ValueError:
                    try:
                        session_date = datetime.strptime(session.timestamp, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        continue
                
                # Filter by date range
                if session_date < start_date or session_date > end_date:
                    continue
                if selected_script is not None and (session.script_name or "") != selected_script:
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
        self._refresh_script_options()
        progress_data = self._get_progress_data()
        self._update_trend_summary()
        
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

    def _format_trouble_words(self, rows: List[dict]) -> str:
        if not rows:
            return "No data for selected period."
        lines = []
        for i, r in enumerate(rows, start=1):
            word = r.get("word", "")
            rate = float(r.get("error_rate", 0.0))
            errs = int(r.get("errors", 0))
            atts = int(r.get("attempts", 0))
            lines.append(f"{i}. {word} - {rate:.0%} ({errs}/{atts})")
        return "\n".join(lines)

    def _format_delta_words(self, rows: List[dict], reverse: bool = False) -> str:
        if not rows:
            return "Not enough prior-window data."
        lines = []
        for i, r in enumerate(rows, start=1):
            word = r.get("word", "")
            delta = float(r.get("delta", 0.0))
            recent = float(r.get("recent_rate", 0.0))
            prev = float(r.get("previous_rate", 0.0))
            arrow = "up" if delta > 0 else "down"
            if reverse:
                arrow = "down" if delta < 0 else "up"
            lines.append(
                f"{i}. {word} - {recent:.0%} vs {prev:.0%} ({arrow} {abs(delta):.0%})"
            )
        return "\n".join(lines)

    def _format_simple_rate_rows(self, rows: List[dict], key: str) -> str:
        if not rows:
            return "No data for selected period."
        lines = []
        for i, r in enumerate(rows, start=1):
            name = str(r.get(key, ""))
            rate = float(r.get("error_rate", 0.0))
            errs = int(r.get("errors", 0))
            atts = int(r.get("attempts", 0))
            lines.append(f"{i}. {name} - {rate:.1%} ({errs}/{atts})")
        return "\n".join(lines)

    def _format_confusions(self, rows: List[dict], top_n: int = 5) -> str:
        if not rows:
            return "No confusion pairs yet."
        lines = []
        for i, r in enumerate(rows[:top_n], start=1):
            src = r.get("from", "?")
            dst = r.get("to", "?")
            cnt = int(r.get("count", 0))
            lines.append(f"{i}. {src} -> {dst} ({cnt})")
        return "\n".join(lines)

    def _update_trend_summary(self):
        trend_labels = [
            self.trouble_words_label,
            self.improved_words_label,
            self.regressed_words_label,
            self.char_kinds_label,
            self.char_confusions_label,
            self.position_label,
            self.position_improved_label,
            self.phoneme_label,
            self.phoneme_confusions_label,
            self.feedback_label,
        ]
        if not hasattr(self.parent_app, "db"):
            for lbl in trend_labels:
                lbl.setText("No DB connection.")
            return
        try:
            start_dt, end_dt, script_name = self._current_filters()
            summary = get_word_trend_summary(
                self.parent_app.db,
                start_dt=start_dt,
                end_dt=end_dt,
                script_name=script_name,
                top_n=5,
                min_attempts=3,
            )
            char_summary = get_character_trend_summary(
                self.parent_app.db,
                start_dt=start_dt,
                end_dt=end_dt,
                script_name=script_name,
                top_n=5,
            )
            pos_summary = get_position_trend_summary(
                self.parent_app.db,
                start_dt=start_dt,
                end_dt=end_dt,
                script_name=script_name,
                top_n=4,
            )
            phon_summary = get_phoneme_trend_summary(
                self.parent_app.db,
                start_dt=start_dt,
                end_dt=end_dt,
                script_name=script_name,
                top_n=5,
                min_attempts=4,
            )
            feedback = generate_feedback_summary(
                summary, char_summary, pos_summary, phon_summary
            )
            self.trouble_words_label.setText(
                self._format_trouble_words(summary.get("top_trouble_words", []))
            )
            self.improved_words_label.setText(
                self._format_delta_words(summary.get("most_improved_words", []))
            )
            self.regressed_words_label.setText(
                self._format_delta_words(
                    summary.get("most_regressed_words", []), reverse=True
                )
            )
            self.char_kinds_label.setText(
                self._format_simple_rate_rows(
                    char_summary.get("top_character_kinds", []), key="kind"
                )
            )
            self.char_confusions_label.setText(
                self._format_confusions(
                    char_summary.get("top_character_confusions", [])
                )
            )
            self.position_label.setText(
                self._format_simple_rate_rows(
                    pos_summary.get("top_position_buckets", []), key="bucket"
                )
            )
            pos_improved = pos_summary.get("most_improved_positions", [])[:2]
            pos_regressed = pos_summary.get("most_regressed_positions", [])[:2]
            pos_lines = []
            for row in pos_improved:
                pos_lines.append(
                    f"improved: {row.get('bucket', '?')} ({abs(float(row.get('delta', 0.0))):.1%})"
                )
            for row in pos_regressed:
                pos_lines.append(
                    f"regressed: {row.get('bucket', '?')} ({abs(float(row.get('delta', 0.0))):.1%})"
                )
            self.position_improved_label.setText(
                "\n".join(pos_lines) if pos_lines else "Not enough prior-window data."
            )
            self.phoneme_label.setText(
                self._format_simple_rate_rows(
                    phon_summary.get("top_trouble_symbols", []), key="symbol"
                )
            )
            self.phoneme_confusions_label.setText(
                self._format_confusions(phon_summary.get("top_symbol_confusions", []))
            )
            self.feedback_label.setText("\n".join(f"- {line}" for line in feedback))
        except Exception as e:
            self.trouble_words_label.setText(f"Trend query failed: {e}")
            for lbl in trend_labels[1:]:
                lbl.setText("Trend query failed.")
    
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
