#!/usr/bin/env python3
"""Perimeter refinement with automatic topology migration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from testing.refine_perimeter_iterative import main
sys.exit(main())
