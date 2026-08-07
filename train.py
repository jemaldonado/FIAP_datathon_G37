#!/usr/bin/env python
"""
Quick training script for Contextual Thompson Sampling model.
Run from project root: python train.py

Wraps scripts/retrain_model.py, the production training path (4-arm
model, 100% real per-arm conversion data). scripts/train_model.py is a
legacy/exploratory script that assigns arms with invented conversion
rates and is not used in production — see its own module docstring.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now run training
from scripts.retrain_model import main

if __name__ == '__main__':
    sys.exit(main())
