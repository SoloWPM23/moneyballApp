"""
Moneyball App - Main Launcher
=============================
Aplikasi analisis pemain sepak bola dengan ML dan AI.

Usage:
    python run_app.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gui.main_window import main

if __name__ == "__main__":
    print("Starting Moneyball App...")
    print("=" * 40)
    main()
