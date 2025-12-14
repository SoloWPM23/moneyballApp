"""
Moneyball App - Main GUI Application
=====================================
Professional GUI for Football Player Analysis using PyQt6.
Integrates ML, Visualization, and AI Storytelling modules.
"""

import sys
import os
import json
import re
from pathlib import Path

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('QtAgg')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QSplitter, QFrame, QGroupBox, QSpinBox,
    QCheckBox, QMessageBox, QStatusBar, QCompleter, QListWidget,
    QGridLayout, QHeaderView, QAbstractItemView, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

# Import our modules
from ml.similarity_model import PlayerSimilarityModel
from visualization.charts import PlayerVisualizer
from ai.gemini_storyteller import GeminiStoryteller, load_api_key

STYLESHEET = """
/* Main Window */
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #eaeaea;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    background-color: #16213e;
    padding: 10px;
}

QTabBar::tab {
    background-color: #1a1a2e;
    color: #8892b0;
    padding: 12px 24px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #16213e;
    color: #64ffda;
    border-bottom: 3px solid #64ffda;
}

QTabBar::tab:hover:!selected {
    background-color: #0f3460;
    color: #ccd6f6;
}

/* Group Box */
QGroupBox {
    font-weight: bold;
    font-size: 14px;
    color: #ccd6f6;
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    margin-top: 16px;
    padding: 15px;
    background-color: #16213e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 8px;
    color: #64ffda;
}

/* Line Edit */
QLineEdit {
    background-color: #0f3460;
    border: 2px solid #3d3d5c;
    border-radius: 6px;
    padding: 10px 15px;
    color: #eaeaea;
    selection-background-color: #64ffda;
    selection-color: #0a192f;
}

QLineEdit:focus {
    border-color: #64ffda;
}

QLineEdit::placeholder {
    color: #5a6785;
}

/* Push Button */
QPushButton {
    background-color: #0f3460;
    color: #ccd6f6;
    border: 2px solid #3d3d5c;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #1a4a7a;
    border-color: #64ffda;
    color: #64ffda;
}

QPushButton:pressed {
    background-color: #0a2540;
}

QPushButton:disabled {
    background-color: #2d2d4a;
    color: #5a6785;
    border-color: #3d3d5c;
}

/* Primary Button */
QPushButton[class="primary"] {
    background-color: #64ffda;
    color: #0a192f;
    border: none;
}

QPushButton[class="primary"]:hover {
    background-color: #4de8c2;
}

/* Danger Button */
QPushButton[class="danger"] {
    background-color: #ff6b6b;
    color: white;
    border: none;
}

QPushButton[class="danger"]:hover {
    background-color: #ee5a5a;
}

/* Combo Box */
QComboBox {
    background-color: #0f3460;
    border: 2px solid #3d3d5c;
    border-radius: 6px;
    padding: 8px 15px;
    color: #eaeaea;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #64ffda;
}

QComboBox::drop-down {
    border: none;
    padding-right: 10px;
}

QComboBox QAbstractItemView {
    background-color: #16213e;
    border: 1px solid #3d3d5c;
    selection-background-color: #0f3460;
    color: #eaeaea;
}

/* Spin Box */
QSpinBox {
    background-color: #0f3460;
    border: 2px solid #3d3d5c;
    border-radius: 6px;
    padding: 8px;
    color: #eaeaea;
}

QSpinBox:focus {
    border-color: #64ffda;
}

/* Table Widget */
QTableWidget {
    background-color: #16213e;
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    gridline-color: #3d3d5c;
    selection-background-color: #0f3460;
}

QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #2d2d4a;
}

QTableWidget::item:selected {
    background-color: #0f3460;
    color: #64ffda;
}

QTableWidget::item:hover {
    background-color: #1a3a5c;
}

QHeaderView::section {
    background-color: #1a1a2e;
    color: #64ffda;
    padding: 12px;
    border: none;
    border-bottom: 2px solid #64ffda;
    font-weight: bold;
}

/* Text Edit */
QTextEdit {
    background-color: #0f3460;
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    padding: 15px;
    color: #ccd6f6;
    line-height: 1.6;
}

/* List Widget */
QListWidget {
    background-color: #0f3460;
    border: 1px solid #3d3d5c;
    border-radius: 6px;
    padding: 5px;
}

QListWidget::item {
    padding: 8px;
    border-radius: 4px;
    margin: 2px;
}

QListWidget::item:selected {
    background-color: #1a4a7a;
    color: #64ffda;
}

/* Scroll Bar */
QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #3d3d5c;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #64ffda;
}

/* Status Bar */
QStatusBar {
    background-color: #1a1a2e;
    color: #8892b0;
    border-top: 1px solid #3d3d5c;
    padding: 5px;
}

/* Label */
QLabel {
    color: #ccd6f6;
}

/* Splitter */
QSplitter::handle {
    background-color: #3d3d5c;
}

QSplitter::handle:hover {
    background-color: #64ffda;
}
"""


# AI WORKER THREAD

class AIWorker(QThread):
    """Worker thread for AI generation to avoid blocking UI."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, storyteller, method_name, *args, **kwargs):
        super().__init__()
        self.storyteller = storyteller
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            method = getattr(self.storyteller, self.method_name)
            result = method(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# EMBEDDED CHART WIDGET

class ChartCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding charts in PyQt."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#16213e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = None
        
    def clear(self):
        """Clear the figure."""
        self.fig.clear()
        self.ax = None
        self.draw()


# MAIN APPLICATION

class MoneyballApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moneyball - Football Player Analysis")
        self.setMinimumSize(1400, 900)
        
        # Paths
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "data" / "dataPlayerC.csv"
        self.config_path = self.base_path / "data" / "config.json"
        
        # Components
        self.df = None
        self.similarity_model = None
        self.visualizer = None
        self.storyteller = None
        self.ai_worker = None
        self.current_recommendations = []
        self.current_target = ""
        
        # Setup
        self.init_ui()
        self.load_data()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Header
        main_layout.addWidget(self.create_header())
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_search_tab(), "Search and Compare")
        self.tabs.addTab(self.create_recommend_tab(), "Recommendations")
        self.tabs.addTab(self.create_analysis_tab(), "Analysis")
        self.tabs.addTab(self.create_ai_tab(), "AI Insights")
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def create_header(self):
        """Create header widget."""
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 12px;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Logo and title
        title = QLabel("MONEYBALL ANALYZER")
        title.setStyleSheet("color: white; font-size: 26px; font-weight: bold;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Stats
        self.stats_label = QLabel("Players: 0 | Leagues: 0 | Teams: 0")
        self.stats_label.setStyleSheet("""
            color: #ccd6f6; 
            font-size: 12px;
            background-color: #0f3460;
            padding: 8px 15px;
            border-radius: 6px;
        """)
        layout.addWidget(self.stats_label)
        
        return header

    def create_search_tab(self):
        """Create search and compare tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Search section
        search_group = QGroupBox("Player Search")
        search_layout = QHBoxLayout(search_group)
        search_layout.setSpacing(10)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter player name (e.g., Mohamed Salah, Haaland)")
        self.search_input.setMinimumWidth(400)
        search_layout.addWidget(self.search_input)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.setProperty("class", "primary")
        self.search_btn.setStyleSheet("""
            background-color: #27ae60; color: white; 
            padding: 12px; font-weight: bold;
        """)
        search_layout.addWidget(self.search_btn)
        search_layout.addStretch()
        
        layout.addWidget(search_group)
        
        # Results splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Results table
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["Player", "Team", "League", "Position", "Goals", "Assists"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget { alternate-background-color: #1a3050; }
        """)
        results_layout.addWidget(self.results_table)
        
        splitter.addWidget(results_group)
        
        # Player details
        details_group = QGroupBox("Player Details")
        details_layout = QVBoxLayout(details_group)
        
        self.player_details = QTextEdit()
        self.player_details.setReadOnly(True)
        self.player_details.setStyleSheet("""
            font-family: 'Consolas', 'Courier New', monospace; 
            font-size: 13px;
            line-height: 1.5;
        """)
        details_layout.addWidget(self.player_details)
        
        # Compare buttons
        btn_layout = QHBoxLayout()
        self.compare_btn = QPushButton("Add to Compare")
        self.compare_btn.setStyleSheet("background-color: #27ae60; color: white;")
        btn_layout.addWidget(self.compare_btn)
        
        self.clear_compare_btn = QPushButton("Clear")
        self.clear_compare_btn.setStyleSheet("background-color: #c0392b; color: white;")
        btn_layout.addWidget(self.clear_compare_btn)
        details_layout.addLayout(btn_layout)
        
        splitter.addWidget(details_group)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        # Compare section
        compare_group = QGroupBox("Player Comparison (max 5)")
        compare_layout = QVBoxLayout(compare_group)
        
        self.compare_list = QListWidget()
        self.compare_list.setMaximumHeight(80)
        self.compare_list.setFlow(QListWidget.Flow.LeftToRight)
        self.compare_list.setStyleSheet("QListWidget::item { margin: 5px; }")
        compare_layout.addWidget(self.compare_list)
        
        self.run_compare_btn = QPushButton("Compare Selected Players")
        self.run_compare_btn.setStyleSheet("""
            background-color: #9b59b6; color: white; 
            padding: 12px; font-weight: bold;
        """)
        compare_layout.addWidget(self.run_compare_btn)
        
        layout.addWidget(compare_group)
        
        return tab
    
    def create_recommend_tab(self):
        """Create recommendations tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Target section
        target_group = QGroupBox("Find Similar Players")
        target_layout = QGridLayout(target_group)
        target_layout.setSpacing(10)
        
        target_layout.addWidget(QLabel("Target Player:"), 0, 0)
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("e.g., Kevin De Bruyne")
        target_layout.addWidget(self.target_input, 0, 1)
        
        target_layout.addWidget(QLabel("Position:"), 0, 2)
        self.position_filter = QComboBox()
        self.position_filter.addItems(["All", "FW", "MF", "DF", "GK"])
        target_layout.addWidget(self.position_filter, 0, 3)
        
        target_layout.addWidget(QLabel("League:"), 1, 0)
        self.league_filter = QComboBox()
        self.league_filter.addItem("All Leagues")
        target_layout.addWidget(self.league_filter, 1, 1)
        
        target_layout.addWidget(QLabel("Top N:"), 1, 2)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 50)
        self.top_n_spin.setValue(10)
        target_layout.addWidget(self.top_n_spin, 1, 3)
        
        self.find_similar_btn = QPushButton("Find Similar Players")
        self.find_similar_btn.setStyleSheet("""
            background-color: #27ae60; color: white; 
            padding: 12px; font-weight: bold;
        """)
        target_layout.addWidget(self.find_similar_btn, 2, 0, 1, 4)
        
        layout.addWidget(target_group)
        
        # Results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Recommendations table
        rec_group = QGroupBox("Similar Players")
        rec_layout = QVBoxLayout(rec_group)
        
        self.rec_table = QTableWidget()
        self.rec_table.setColumnCount(7)
        self.rec_table.setHorizontalHeaderLabels([
            "Player", "Team", "League", "Position", "Goals", "Assists", "Similarity"
        ])
        self.rec_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rec_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rec_table.setAlternatingRowColors(True)
        self.rec_table.setStyleSheet("QTableWidget { alternate-background-color: #1a3050; }")
        rec_layout.addWidget(self.rec_table)
        
        splitter.addWidget(rec_group)
        
        # Explanation
        explain_group = QGroupBox("AI Explanation")
        explain_layout = QVBoxLayout(explain_group)
        
        self.explain_text = QTextEdit()
        self.explain_text.setReadOnly(True)
        explain_layout.addWidget(self.explain_text)
        
        self.explain_btn = QPushButton("Generate AI Explanation")
        self.explain_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        explain_layout.addWidget(self.explain_btn)
        
        splitter.addWidget(explain_group)
        splitter.setSizes([550, 450])
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_analysis_tab(self):
        """Create analysis/visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Options
        options_group = QGroupBox("Visualization Options")
        options_layout = QGridLayout(options_group)
        options_layout.setSpacing(10)
        
        options_layout.addWidget(QLabel("Player 1:"), 0, 0)
        self.analysis_player1 = QLineEdit()
        self.analysis_player1.setPlaceholderText("Required - Enter player name")
        options_layout.addWidget(self.analysis_player1, 0, 1)
        
        options_layout.addWidget(QLabel("Player 2:"), 0, 2)
        self.analysis_player2 = QLineEdit()
        self.analysis_player2.setPlaceholderText("Optional - For comparison")
        options_layout.addWidget(self.analysis_player2, 0, 3)
        
        options_layout.addWidget(QLabel("Chart Type:"), 1, 0)
        self.chart_type = QComboBox()
        self.chart_type.addItems(["Radar Chart", "Bar Comparison", "Scatter Plot", "Stats Table"])
        options_layout.addWidget(self.chart_type, 1, 1)
        
        self.generate_chart_btn = QPushButton("Generate Visualization")
        self.generate_chart_btn.setStyleSheet("""
            background-color: #27ae60; color: white; 
            padding: 12px; font-weight: bold;
        """)
        options_layout.addWidget(self.generate_chart_btn, 1, 2, 1, 2)
        
        layout.addWidget(options_group)
        
        # Chart display
        chart_group = QGroupBox("Visualization")
        chart_layout = QVBoxLayout(chart_group)
        
        # Scroll area for chart
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setStyleSheet("""
            QScrollArea { 
                border: none; 
                background-color: #16213e;
            }
            QScrollBar:vertical {
                background-color: #1a1a2e;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #3d3d5c;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar:horizontal {
                background-color: #1a1a2e;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background-color: #3d3d5c;
                border-radius: 6px;
                min-width: 30px;
            }
        """)
        
        # Embedded chart canvas
        self.chart_canvas = ChartCanvas(self, width=12, height=8, dpi=100)
        self.chart_canvas.setMinimumSize(800, 600)
        self.chart_scroll.setWidget(self.chart_canvas)
        chart_layout.addWidget(self.chart_scroll)
        
        # Stats table (hidden by default)
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setVisible(False)
        self.stats_display.setMaximumHeight(300)
        self.stats_display.setStyleSheet("font-family: 'Consolas', monospace;")
        chart_layout.addWidget(self.stats_display)
        
        layout.addWidget(chart_group)
        
        return tab
    
    def create_ai_tab(self):
        """Create AI insights tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # AI Analysis
        ai_group = QGroupBox("AI Player Analysis")
        ai_layout = QGridLayout(ai_group)
        ai_layout.setSpacing(10)
        
        ai_layout.addWidget(QLabel("Player:"), 0, 0)
        self.ai_player_input = QLineEdit()
        self.ai_player_input.setPlaceholderText("Enter player name")
        ai_layout.addWidget(self.ai_player_input, 0, 1)
        
        ai_layout.addWidget(QLabel("Type:"), 0, 2)
        self.ai_type = QComboBox()
        self.ai_type.addItems(["Player Profile", "Scout Report", "Quick Summary"])
        ai_layout.addWidget(self.ai_type, 0, 3)
        
        self.generate_ai_btn = QPushButton("Generate AI Analysis")
        self.generate_ai_btn.setStyleSheet("""
            background-color: #27ae60; color: white; 
            padding: 12px; font-weight: bold;
        """)
        ai_layout.addWidget(self.generate_ai_btn, 1, 0, 1, 4)
        
        layout.addWidget(ai_group)
        
        # AI Comparison
        compare_group = QGroupBox("AI Player Comparison")
        compare_layout = QGridLayout(compare_group)
        compare_layout.setSpacing(10)
        
        compare_layout.addWidget(QLabel("Player 1:"), 0, 0)
        self.ai_compare_player1 = QLineEdit()
        compare_layout.addWidget(self.ai_compare_player1, 0, 1)
        
        compare_layout.addWidget(QLabel("Player 2:"), 0, 2)
        self.ai_compare_player2 = QLineEdit()
        compare_layout.addWidget(self.ai_compare_player2, 0, 3)
        
        self.ai_compare_btn = QPushButton("Compare with AI")
        self.ai_compare_btn.setStyleSheet("""
            background-color: #27ae60; color: white; 
            padding: 12px; font-weight: bold;
        """)
        compare_layout.addWidget(self.ai_compare_btn, 1, 0, 1, 4)
        
        layout.addWidget(compare_group)
        
        # Output
        output_group = QGroupBox("AI Analysis Output")
        output_layout = QVBoxLayout(output_group)
        
        self.ai_output = QTextEdit()
        self.ai_output.setReadOnly(True)
        self.ai_output.setStyleSheet("""
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            line-height: 1.8;
            padding: 15px;
        """)
        output_layout.addWidget(self.ai_output)
        
        self.ai_progress = QLabel("")
        self.ai_progress.setStyleSheet("color: #64ffda; font-style: italic;")
        output_layout.addWidget(self.ai_progress)
        
        layout.addWidget(output_group)
        
        return tab

    def load_data(self):
        """Load data and initialize components."""
        try:
            self.statusBar.showMessage("Loading data...")
            
            # Load dataset
            self.df = pd.read_csv(self.data_path)
            
            # Add main position
            if 'Pos_Main' not in self.df.columns:
                self.df['Pos_Main'] = self.df['Pos'].apply(
                    lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown'
                )
            
            # Fill NaN
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
            
            # Initialize models
            self.similarity_model = PlayerSimilarityModel(self.df)
            self.similarity_model.prepare_data()
            self.similarity_model.compute_similarity_matrix()
            
            self.visualizer = PlayerVisualizer(self.df)
            
            # Initialize AI
            try:
                api_key = load_api_key(str(self.config_path))
                if api_key:
                    self.storyteller = GeminiStoryteller(api_key, self.df)
                    self.statusBar.showMessage("All systems ready")
                else:
                    self.statusBar.showMessage("No API key - AI features disabled")
            except Exception as e:
                self.statusBar.showMessage(f"AI error: {str(e)[:30]}")
            
            # Update UI
            self.update_stats()
            self.setup_autocomplete()
            self.populate_leagues()
            
            self.statusBar.showMessage(f"Loaded {len(self.df)} players successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            
    def setup_connections(self):
        """Setup signal connections."""
        # Search tab
        self.search_btn.clicked.connect(self.search_players)
        self.search_input.returnPressed.connect(self.search_players)
        self.results_table.itemSelectionChanged.connect(self.show_player_details)
        self.compare_btn.clicked.connect(self.add_to_compare)
        self.clear_compare_btn.clicked.connect(self.clear_compare)
        self.run_compare_btn.clicked.connect(self.run_comparison)
        
        # Recommend tab
        self.find_similar_btn.clicked.connect(self.find_similar_players)
        self.explain_btn.clicked.connect(self.generate_recommendation_explanation)
        
        # Analysis tab
        self.generate_chart_btn.clicked.connect(self.generate_chart)
        
        # AI tab
        self.generate_ai_btn.clicked.connect(self.generate_ai_analysis)
        self.ai_compare_btn.clicked.connect(self.generate_ai_comparison)
        
    def update_stats(self):
        """Update stats label."""
        if self.df is not None:
            n_players = len(self.df)
            n_leagues = self.df['Comp'].nunique()
            n_teams = self.df['Squad'].nunique()
            self.stats_label.setText(f"Players: {n_players:,} | Leagues: {n_leagues} | Teams: {n_teams}")
            
    def setup_autocomplete(self):
        """Setup autocomplete."""
        if self.df is not None:
            players = self.df['Player'].tolist()
            for widget in [self.search_input, self.target_input, 
                          self.analysis_player1, self.analysis_player2,
                          self.ai_player_input, self.ai_compare_player1, 
                          self.ai_compare_player2]:
                completer = QCompleter(players)
                completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                completer.setMaxVisibleItems(10)
                widget.setCompleter(completer)
                
    def populate_leagues(self):
        """Populate league filter."""
        if self.df is not None:
            leagues = sorted(self.df['Comp'].unique())
            self.league_filter.addItems(leagues)
            
    def search_players(self):
        """Search for players."""
        query = self.search_input.text().strip()
        if not query:
            return
        
        mask = self.df['Player'].str.contains(query, case=False, na=False)
        results = self.df[mask].head(25)
        
        self.results_table.setRowCount(len(results))
        for i, (_, row) in enumerate(results.iterrows()):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(row['Player'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(row['Squad'])))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(row['Comp'])))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(row['Pos'])))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(int(row.get('Gls', 0)))))
            self.results_table.setItem(i, 5, QTableWidgetItem(str(int(row.get('Ast', 0)))))
        
        self.statusBar.showMessage(f"Found {len(results)} players matching '{query}'")
        
    def show_player_details(self):
        """Show selected player details."""
        row = self.results_table.currentRow()
        if row < 0:
            return
        
        item = self.results_table.item(row, 0)
        if not item:
            return
        
        name = item.text()
        matches = self.df[self.df['Player'] == name]
        if len(matches) == 0:
            matches = self.df[self.df['Player'].str.contains(name, case=False, na=False)]
        
        if len(matches) == 0:
            return
        
        p = matches.iloc[0]
        
        details = f"""
═══════════════ { 'PLAYER PROFILE' } ═══════════════
Player:
{p['Player']:<40}

═══════════════ PLAYER DETAILS ═══════════════

Team:      {p['Squad']}
League:    {p['Comp']}
Position:  {p['Pos']}

═══════════════ OFFENSIVE ═══════════════

Goals:       {p.get('Gls', 0):<8}   xG:   {p.get('xG', 0):.2f}
Assists:    {p.get('Ast', 0):<8}   xAG:  {p.get('xAG', 0):.2f}
G+A:        {p.get('G+A', 0):<8}

Shots:      {p.get('Sh', 0):<8}    SoT:  {p.get('SoT', 0)}

═══════════════ PROGRESSION ═══════════════

Prog.Carries:   {p.get('PrgC', 0)}
Prog.Passes:    {p.get('PrgP', 0)}

═══════════════ DEFENSIVE ═══════════════

Tackles:       {p.get('Tkl', 0)}
Interceptions:  {p.get('Int', 0)}

═══════════════ PLAYING TIME ═══════════════

Minutes:  {p.get('Min', 0):,}
"""
        self.player_details.setText(details)
        
    def add_to_compare(self):
        """Add player to compare list."""
        row = self.results_table.currentRow()
        if row < 0:
            return
        
        item = self.results_table.item(row, 0)
        if not item:
            return
        
        name = item.text()
        
        # Check duplicates
        for i in range(self.compare_list.count()):
            if self.compare_list.item(i).text() == name:
                self.statusBar.showMessage(f"'{name}' already in compare list")
                return
        
        if self.compare_list.count() >= 5:
            QMessageBox.warning(self, "Limit", "Maximum 5 players for comparison")
            return
        
        self.compare_list.addItem(name)
        self.statusBar.showMessage(f"Added '{name}' to comparison")
        
    def clear_compare(self):
        """Clear compare list."""
        self.compare_list.clear()
        self.statusBar.showMessage("Compare list cleared")
        
    def run_comparison(self):
        """Run player comparison with chart."""
        if self.compare_list.count() < 2:
            QMessageBox.warning(self, "Warning", "Select at least 2 players")
            return
        
        players = [self.compare_list.item(i).text() for i in range(self.compare_list.count())]
        
        # Build comparison text
        text = "═══════════════ PLAYER COMPARISON ═══════════════\n\n"
        text += f"{'Player':<20} {'Team':<15} {'Gls':>5} {'Ast':>5} {'G+A':>5} {'xG':>6}\n"
        text += "─" * 60 + "\n"
        
        for name in players:
            p = self.df[self.df['Player'] == name]
            if len(p) == 0:
                p = self.df[self.df['Player'].str.contains(name, case=False, na=False)]
            if len(p) > 0:
                p = p.iloc[0]
                text += f"{name[:20]:<20} {str(p['Squad'])[:15]:<15} "
                text += f"{int(p.get('Gls', 0)):>5} {int(p.get('Ast', 0)):>5} "
                text += f"{int(p.get('G+A', 0)):>5} {p.get('xG', 0):>6.2f}\n"
        
        text += "\nTip: Go to Analysis tab for visual comparison charts"
        
        self.player_details.setText(text)
        
        # Auto-fill Analysis tab
        if len(players) >= 2:
            self.analysis_player1.setText(players[0])
            self.analysis_player2.setText(players[1])
        
        self.statusBar.showMessage("Comparison ready - check Analysis tab for charts")
        
    def find_similar_players(self):
        """Find similar players."""
        target = self.target_input.text().strip()
        if not target:
            QMessageBox.warning(self, "Warning", "Enter a player name")
            return
        
        matches = self.df[self.df['Player'].str.contains(target, case=False, na=False)]
        if len(matches) == 0:
            QMessageBox.warning(self, "Not Found", f"Player '{target}' not found")
            return
        
        target_name = matches.iloc[0]['Player']
        position = self.position_filter.currentText()
        league = self.league_filter.currentText()
        top_n = self.top_n_spin.value()
        
        try:
            similar_df = self.similarity_model.get_similar_players(target_name, top_n=top_n + 10)
            
            if similar_df.empty:
                QMessageBox.warning(self, "Not Found", "No similar players found")
                return
            
            # Apply filters
            if position != "All":
                similar_df = similar_df[similar_df['Pos'].str.contains(position, case=False, na=False)]
            if league != "All Leagues":
                similar_df = similar_df[similar_df['Comp'] == league]
            
            similar_df = similar_df.head(top_n)
            
            # Build recommendations list
            similar = []
            for _, row in similar_df.iterrows():
                player_data = self.df[self.df['Player'] == row['Player']]
                if len(player_data) > 0:
                    player_data = player_data.iloc[0]
                    similar.append({
                        'name': row['Player'],
                        'squad': row['Squad'],
                        'league': row['Comp'],
                        'position': row['Pos'],
                        'goals': int(player_data.get('Gls', 0)),
                        'assists': int(player_data.get('Ast', 0)),
                        'similarity_score': row['Similarity']
                    })
            
            self.current_recommendations = similar
            self.current_target = target_name
            
            # Populate table
            self.rec_table.setRowCount(len(similar))
            for i, p in enumerate(similar):
                self.rec_table.setItem(i, 0, QTableWidgetItem(p['name']))
                self.rec_table.setItem(i, 1, QTableWidgetItem(p['squad']))
                self.rec_table.setItem(i, 2, QTableWidgetItem(p['league']))
                self.rec_table.setItem(i, 3, QTableWidgetItem(p['position']))
                self.rec_table.setItem(i, 4, QTableWidgetItem(str(p['goals'])))
                self.rec_table.setItem(i, 5, QTableWidgetItem(str(p['assists'])))
                self.rec_table.setItem(i, 6, QTableWidgetItem(f"{p['similarity_score']:.1%}"))
            
            self.statusBar.showMessage(f"Found {len(similar)} similar players to {target_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def generate_recommendation_explanation(self):
        """Generate AI explanation."""
        if not self.current_recommendations:
            QMessageBox.warning(self, "Warning", "Find similar players first")
            return
        
        if not self.storyteller:
            QMessageBox.warning(self, "Warning", "AI not available")
            return
        
        self.explain_text.setText("Generating AI explanation...")
        self.explain_btn.setEnabled(False)
        
        self.ai_worker = AIWorker(
            self.storyteller, 'generate_recommendation_explanation',
            self.current_target, self.current_recommendations
        )
        self.ai_worker.finished.connect(self.on_explanation_ready)
        self.ai_worker.error.connect(self.on_ai_error)
        self.ai_worker.start()
        
    def on_explanation_ready(self, result):
        """Handle explanation result."""
        self.explain_text.setText(self.clean_markdown(result))
        self.explain_btn.setEnabled(True)

    def generate_chart(self):
        """Generate visualization."""
        player1 = self.analysis_player1.text().strip()
        player2 = self.analysis_player2.text().strip()
        chart_type = self.chart_type.currentText()
        
        if not player1:
            QMessageBox.warning(self, "Warning", "Enter at least Player 1")
            return
        
        # Find players
        p1_match = self.df[self.df['Player'].str.contains(player1, case=False, na=False)]
        if len(p1_match) == 0:
            QMessageBox.warning(self, "Not Found", f"Player '{player1}' not found")
            return
        
        p1 = p1_match.iloc[0]
        p2 = None
        
        if player2:
            p2_match = self.df[self.df['Player'].str.contains(player2, case=False, na=False)]
            if len(p2_match) > 0:
                p2 = p2_match.iloc[0]
        
        # Generate chart
        self.stats_display.setVisible(False)
        self.chart_scroll.setVisible(True)
        self.chart_canvas.setVisible(True)
        
        if chart_type == "Stats Table":
            self.show_stats_table(p1, p2)
        elif chart_type == "Radar Chart":
            self.show_radar_chart(p1, p2)
        elif chart_type == "Bar Comparison":
            self.show_bar_comparison(p1, p2)
        elif chart_type == "Scatter Plot":
            self.show_scatter_plot(p1, p2)
            
    def show_radar_chart(self, p1, p2=None):
        """Show radar chart embedded in canvas."""
        self.chart_canvas.fig.clear()
        
        stats = ['Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'Tkl', 'Int']
        labels = ['Goals', 'Assists', 'xG', 'xAG', 'Prog.Car', 'Prog.Pas', 'Tackles', 'Int']
        
        # Normalize values
        values1 = []
        for s in stats:
            val = p1.get(s, 0)
            max_val = self.df[s].max() if s in self.df.columns else 1
            values1.append((val / max_val * 100) if max_val > 0 else 0)
        values1 += values1[:1]
        
        angles = [n / float(len(stats)) * 2 * pi for n in range(len(stats))]
        angles += angles[:1]
        
        ax = self.chart_canvas.fig.add_subplot(111, polar=True, facecolor='#16213e')
        ax.set_facecolor('#16213e')
        
        ax.plot(angles, values1, 'o-', linewidth=2, label=p1['Player'], color='#64ffda')
        ax.fill(angles, values1, alpha=0.25, color='#64ffda')
        
        if p2 is not None:
            values2 = []
            for s in stats:
                val = p2.get(s, 0)
                max_val = self.df[s].max() if s in self.df.columns else 1
                values2.append((val / max_val * 100) if max_val > 0 else 0)
            values2 += values2[:1]
            ax.plot(angles, values2, 'o-', linewidth=2, label=p2['Player'], color='#ff6b6b')
            ax.fill(angles, values2, alpha=0.25, color='#ff6b6b')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='#ccd6f6', size=9)
        ax.tick_params(colors='#8892b0')
        ax.grid(color='#3d3d5c', alpha=0.5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                  facecolor='#16213e', edgecolor='#3d3d5c', labelcolor='#ccd6f6')
        
        title = f"{p1['Player']}" + (f" vs {p2['Player']}" if p2 is not None else "")
        ax.set_title(title, color='#64ffda', size=14, fontweight='bold', y=1.1)
        
        self.chart_canvas.fig.tight_layout()
        self.chart_canvas.draw()
        self.statusBar.showMessage(f"Radar Chart: {title}")
        
    def show_bar_comparison(self, p1, p2=None):
        """Show bar chart embedded in canvas."""
        self.chart_canvas.fig.clear()
        
        stats = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'Sh', 'SoT']
        labels = ['Goals', 'Assists', 'G+A', 'xG', 'xAG', 'Shots', 'SoT']
        
        values1 = [p1.get(s, 0) for s in stats]
        x = np.arange(len(stats))
        width = 0.35
        
        ax = self.chart_canvas.fig.add_subplot(111, facecolor='#16213e')
        
        has_p2 = p2 is not None
        bars1 = ax.bar(x - width/2 if has_p2 else x, values1, width, 
                       label=p1['Player'], color='#64ffda', edgecolor='#4de8c2')
        
        if has_p2:
            values2 = [p2.get(s, 0) for s in stats]
            bars2 = ax.bar(x + width/2, values2, width, 
                          label=p2['Player'], color='#ff6b6b', edgecolor='#ee5a5a')
        
        ax.set_ylabel('Value', color='#ccd6f6')
        ax.set_xlabel('Statistics', color='#ccd6f6')
        ax.set_title('Player Statistics Comparison', color='#64ffda', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', color='#ccd6f6')
        ax.tick_params(colors='#8892b0')
        ax.legend(facecolor='#16213e', edgecolor='#3d3d5c', labelcolor='#ccd6f6')
        ax.set_facecolor('#16213e')
        ax.spines['bottom'].set_color('#3d3d5c')
        ax.spines['left'].set_color('#3d3d5c')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#3d3d5c', alpha=0.3)
        
        ax.bar_label(bars1, padding=3, color='#ccd6f6', fontsize=8)
        if has_p2:
            ax.bar_label(bars2, padding=3, color='#ccd6f6', fontsize=8)
        
        self.chart_canvas.fig.tight_layout()
        self.chart_canvas.draw()
        self.statusBar.showMessage(f"Bar Chart: {p1['Player']}" + (f" vs {p2['Player']}" if has_p2 else ""))
        
    def show_scatter_plot(self, p1, p2=None):
        """Show scatter plot embedded in canvas."""
        self.chart_canvas.fig.clear()
        
        ax = self.chart_canvas.fig.add_subplot(111, facecolor='#16213e')
        
        # All players
        ax.scatter(self.df['Gls'], self.df['Ast'], alpha=0.3, c='#5a6785', s=30, label='All Players')
        
        # Player 1
        ax.scatter(p1.get('Gls', 0), p1.get('Ast', 0), c='#64ffda', s=200,
                   label=p1['Player'], edgecolors='white', linewidth=2, zorder=5)
        ax.annotate(p1['Player'], (p1.get('Gls', 0), p1.get('Ast', 0)),
                    textcoords="offset points", xytext=(10, 10), color='#64ffda', fontsize=10)
        
        # Player 2
        if p2 is not None:
            ax.scatter(p2.get('Gls', 0), p2.get('Ast', 0), c='#ff6b6b', s=200,
                       label=p2['Player'], edgecolors='white', linewidth=2, zorder=5)
            ax.annotate(p2['Player'], (p2.get('Gls', 0), p2.get('Ast', 0)),
                        textcoords="offset points", xytext=(10, 10), color='#ff6b6b', fontsize=10)
        
        ax.set_xlabel('Goals', color='#ccd6f6')
        ax.set_ylabel('Assists', color='#ccd6f6')
        ax.set_title('Goals vs Assists Distribution', color='#64ffda', fontweight='bold')
        ax.legend(facecolor='#16213e', edgecolor='#3d3d5c', labelcolor='#ccd6f6')
        ax.tick_params(colors='#8892b0')
        ax.set_facecolor('#16213e')
        ax.spines['bottom'].set_color('#3d3d5c')
        ax.spines['left'].set_color('#3d3d5c')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, color='#3d3d5c')
        
        self.chart_canvas.fig.tight_layout()
        self.chart_canvas.draw()
        self.statusBar.showMessage(f"Scatter Plot: {p1['Player']}" + (f" vs {p2['Player']}" if p2 is not None else ""))
        
    def show_stats_table(self, p1, p2=None):
        """Show stats table."""
        self.chart_scroll.setVisible(False)
        self.chart_canvas.setVisible(False)
        self.stats_display.setVisible(True)
        
        stats = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'Sh', 'SoT', 'PrgC', 'PrgP', 'Tkl', 'Int', 'Min']
        labels = ['Goals', 'Assists', 'G+A', 'xG', 'xAG', 'Shots', 'SoT', 'Prog.Carries', 
                  'Prog.Passes', 'Tackles', 'Interceptions', 'Minutes']
        
        text = "\n" + "═" * 60 + "\n"
        text += f"{'STATISTICS':<20} {p1['Player']:<18}"
        if p2 is not None:
            text += f" {p2['Player']:<18}"
        text += "\n" + "═" * 60 + "\n"
        
        for stat, label in zip(stats, labels):
            val1 = p1.get(stat, 0)
            if isinstance(val1, float):
                val1_str = f"{val1:.2f}"
            else:
                val1_str = str(int(val1))
            
            text += f"{label:<20} {val1_str:<18}"
            
            if p2 is not None:
                val2 = p2.get(stat, 0)
                if isinstance(val2, float):
                    val2_str = f"{val2:.2f}"
                else:
                    val2_str = str(int(val2))
                text += f" {val2_str:<18}"
            text += "\n"
        
        text += "═" * 60 + "\n"
        self.stats_display.setText(text)
        self.statusBar.showMessage(f"Stats Table: {p1['Player']}" + (f" vs {p2['Player']}" if p2 is not None else ""))

    def generate_ai_analysis(self):
        """Generate AI analysis."""
        name = self.ai_player_input.text().strip()
        analysis_type = self.ai_type.currentText()
        
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a player name")
            return
        
        if not self.storyteller:
            QMessageBox.warning(self, "Warning", "AI not available. Check API key.")
            return
        
        self.ai_output.setText("")
        self.ai_progress.setText("Generating AI analysis... Please wait...")
        self.generate_ai_btn.setEnabled(False)
        
        method_map = {
            "Player Profile": "generate_player_description",
            "Scout Report": "generate_scout_report",
            "Quick Summary": "quick_summary"
        }
        
        method = method_map.get(analysis_type, "generate_player_description")
        
        self.ai_worker = AIWorker(self.storyteller, method, name)
        self.ai_worker.finished.connect(self.on_ai_analysis_ready)
        self.ai_worker.error.connect(self.on_ai_error)
        self.ai_worker.start()
        
    def on_ai_analysis_ready(self, result):
        """Handle AI result."""
        self.ai_output.setText(self.clean_markdown(result))
        self.ai_progress.setText("Analysis complete")
        self.generate_ai_btn.setEnabled(True)
        
    def generate_ai_comparison(self):
        """Generate AI comparison."""
        p1 = self.ai_compare_player1.text().strip()
        p2 = self.ai_compare_player2.text().strip()
        
        if not p1 or not p2:
            QMessageBox.warning(self, "Warning", "Enter both player names")
            return
        
        if not self.storyteller:
            QMessageBox.warning(self, "Warning", "AI not available")
            return
        
        self.ai_output.setText("")
        self.ai_progress.setText("Generating AI comparison... Please wait...")
        self.ai_compare_btn.setEnabled(False)
        
        self.ai_worker = AIWorker(self.storyteller, 'generate_comparison_narrative', p1, p2)
        self.ai_worker.finished.connect(self.on_ai_comparison_ready)
        self.ai_worker.error.connect(self.on_ai_error)
        self.ai_worker.start()
        
    def on_ai_comparison_ready(self, result):
        """Handle AI comparison result."""
        self.ai_output.setText(self.clean_markdown(result))
        self.ai_progress.setText("Comparison complete")
        self.ai_compare_btn.setEnabled(True)
        
    def on_ai_error(self, error):
        """Handle AI error."""
        self.ai_output.setText(f"Error: {error}")
        self.ai_progress.setText("")
        self.generate_ai_btn.setEnabled(True)
        self.ai_compare_btn.setEnabled(True)
        self.explain_btn.setEnabled(True)
        
    def clean_markdown(self, text):
        """Clean markdown formatting."""
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# MAIN ENTRY POINT

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(STYLESHEET)
    
    window = MoneyballApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
