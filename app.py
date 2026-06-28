"""
UIT RAG System - Entry point for Streamlit app.
Run: streamlit run app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.app import main

if __name__ == "__main__":
    main()
