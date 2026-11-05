#!/usr/bin/env python3
"""Run the bundled direct evaluator; model and simulator dependencies are required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.scripts.evaluate_libero import main

if __name__ == '__main__':
    main()
