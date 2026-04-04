# Debug Archive

This directory contains test scripts created during the investigation and fix of the fragmentation bug (October 2025).

## Contents

### Phase Validation Scripts
- **`test_phase1_triangle_segments.py`** - Validated Phase 1 triangle-based structures
- **`test_phase2_perimeter_calculator.py`** - Validated Phase 2 PerimeterCalculator refactoring

### Diagnostic Scripts
- **`test_variable_points.py`** - Analyzed variable point distribution
- **`test_refined_contour_extraction.py`** - Validated triangle-based contour extraction
- **`test_lambda_values.py`** - Analyzed λ parameter behavior
- **`test_initial_partitions.py`** - Validated initial partition setup

## Status

All scripts served their purpose and validation is complete. The bug has been fixed through:
- **Phase 1:** Added triangle-based structures
- **Phase 2:** Refactored PerimeterCalculator
- **Phase 3:** Rewrote to_visualization_format()
- **Phase 4:** Removed deprecated CellContour code

## Usage

These scripts can still be run for reference or if similar issues arise:

```bash
# Example: Run Phase 1 validation
python scripts/debug_archive/test_phase1_triangle_segments.py <solution.h5>

# Example: Run Phase 2 validation
python scripts/debug_archive/test_phase2_perimeter_calculator.py
```

## Historical Context

These scripts documented the evolution from ordering-based (broken) to triangle-based (working) segment extraction. They show:
1. How the bug was identified
2. How each phase was validated
3. The comparison between old and new methods

Kept for historical reference and future debugging scenarios.

