# Logging Improvements Summary

**Date**: March 2, 2026  
**Purpose**: Implement verbosity control and correct logging severity levels

---

## Overview

The logging system has been refactored to:
1. **Add debug mode control** via `--debug` flag
2. **Reclassify logging levels** to distinguish expected filtering from actual errors
3. **Add high-level summaries** for normal operation
4. **Keep true errors** labeled as ERROR

---

## Changes Made

### 1. Command-Line Interface (2 files)

#### `testing/test_migration_and_continue.py`
- ✅ Added `--debug` flag (overrides `--log-level`)
- ✅ Dynamic log level configuration based on flag
- ✅ Changed migration failure messages from WARNING → ERROR (lines 176-183, 290-296)

#### `testing/refine_perimeter_iterative.py`
- ✅ Added `--debug` flag (overrides `--log-level`)
- ✅ Dynamic log level configuration based on flag

### 2. Core Analysis Module (1 file)

#### `src/core/type1_component_analyzer.py`

**Reclassified to DEBUG (expected filtering):**
- VP validation failures (lines 379, 426, 583)
- Triplet validation failures (line 597, 599)
- No valid triplet/auxiliary errors (lines 606, 440)
- Component exclusions (line 1026, 1012)
- Type 2 protection neighbor detection warnings (lines 1306, 1330)

**New INFO-level summaries (high-level progress):**
- Component detection summary (line 1201)
- Conflict detection summary (line 1214)
- Type 2 protection summary (line 1222)
- Selection summary (lines 1227-1231)

**Kept as WARNING/ERROR (unexpected issues):**
- Chain detection warnings (line 844)
- Target vertex identification failures (line 100, 280, 550, 557)

---

## Usage

### Normal Mode (Default - Clean Output)
```bash
# Shows high-level summaries only
python testing/test_migration_and_continue.py --solution file.h5 --iterations 5

# Example output:
✓ Found 113 connected component(s)
Type 2 Protection: Processing 8 triple points
Type 2 Protection: Found 48 protected VPs, excluded 15 components
Component selection complete:
  Total detected: 113
  Available for migration: 7
  Excluded (Type 2 protection): 15
  Excluded (other): 91
✓ Component 25: Migration successful
```

### Debug Mode (Verbose - Full Diagnostics)
```bash
# Shows all detailed validation steps
python testing/test_migration_and_continue.py --solution file.h5 --iterations 5 --debug

# Example output includes:
Component 0:     ✗ VP 1540: edge (36918, 36919), λ=0.129631 approaches 36919, NOT 37111
Component 0:   ✗ Triplet 'left-left-center' is INVALID - 2 VP(s) don't approach target
Component 0: Excluded - Cannot find valid triplet
... (all VP-level details)
```

---

## Logging Level Classification

### 🔍 DEBUG (10) - Diagnostic Details
**Question**: "Is this showing HOW the algorithm works internally?"

**Examples:**
- Individual VP approach checks
- Triplet validation attempts
- Lambda values and edge endpoints
- Distance calculations
- Component exclusion reasons

**Visibility:**
- Hidden in normal mode
- Shown with `--debug` flag

---

### ℹ️ INFO (20) - High-Level Progress
**Question**: "Would a user want to know this happened during normal operation?"

**Examples:**
- "Found 113 connected components"
- "Type 2 Protection: 48 protected VPs"
- "Component selection complete: 7 available"
- "✓ Migration successful"

**Visibility:**
- Always shown (default)

---

### ⚠️ WARNING (30) - Unexpected but Recoverable
**Question**: "Is this unusual but the program can continue?"

**Examples:**
- Component chain detection
- Unexpected topology (e.g., triple point has 4 VPs)
- No valid candidates found (unusual case)

**Visibility:**
- Always shown

---

### ❌ ERROR (40) - Operation Failed
**Question**: "Did something that was SUPPOSED to work actually fail?"

**Examples:**
- Migration failed after selection
- File I/O failures
- Topology corruption
- Optimization crashed

**Visibility:**
- Always shown

---

## Key Distinctions

### ❌ NOT an Error (Changed to DEBUG)
```python
# BEFORE (incorrect):
logger.error("Component 0: No valid triplet found")

# AFTER (correct):
logger.debug("Component 0: Excluded - no valid triplet")
```
**Reason**: This is **expected filtering** during candidate validation.

### ✅ IS an Error (Changed from WARNING)
```python
# BEFORE (incorrect):
logger.warning(f"✗ Component {idx}: Migration FAILED")

# AFTER (correct):
logger.error(f"✗ Component {idx}: Migration FAILED")
```
**Reason**: Migration was **selected and attempted** but **failed** — requires attention.

---

## Decision Framework

### The Critical Question:
**"When did this happen in the pipeline?"**

```
1. DETECT candidates                     
2. VALIDATE/FILTER candidates  ← Rejections here = DEBUG
3. SELECT best candidate                 
4. EXECUTE migration          ← Failures here = ERROR
```

- **During validation (step 2):** Component doesn't qualify → DEBUG
- **During execution (step 4):** Migration fails → ERROR

---

## Testing

### Test Normal Mode
```bash
python testing/test_migration_and_continue.py \
  --solution examples/base_solution_iteration1_refined_contours.h5 \
  --iterations 1

# Expected: Clean, high-level output with ~20 lines per iteration
```

### Test Debug Mode
```bash
python testing/test_migration_and_continue.py \
  --solution examples/base_solution_iteration1_refined_contours.h5 \
  --iterations 1 \
  --debug

# Expected: Verbose output with ~1000+ lines showing all VP validations
```

### Verify Error Classification
```bash
# Check that true migration failures show as ERROR
python testing/test_migration_and_continue.py \
  --solution examples/problematic_iteration.h5 \
  --iterations 1 | grep ERROR

# Should show ERROR messages ONLY for actual migration failures
```

---

## Benefits

✅ **Cleaner default output**: Users see progress, not diagnostics  
✅ **Debugging capability**: `--debug` provides full details when needed  
✅ **Correct severity**: Errors are actual failures, not expected filtering  
✅ **Better troubleshooting**: Easy to spot real problems  
✅ **Standard practice**: Follows Python scientific computing conventions  

---

## Output Comparison

### Normal Mode (Clean):
```
Type 2 Protection: 8 triple points, 48 protected VPs, 113 components analyzed
Type 2 Protection: 105 available, 8 excluded
Component selection complete:
  Total detected: 113
  Available for migration: 7
```
**~20-30 lines per iteration**

### Debug Mode (Verbose):
```
Type 2 Protection: 8 triple points, 48 protected VPs, 113 components analyzed
  Processing triple point: VPs [231, 229, 232]
    VP 231 → first_level: 230
    VP 231 → second_level: 228
    ... (all VP calculations)
  Protected VPs: [214, 215, 228, 230, ...]
Component 25: ✓ Auxiliary component for 2-VP [92, 93]: [92, 93, 94]
  ... (all component details)
Type 2 Protection: 105 available, 8 excluded
```
**~1000+ lines per iteration**

---

## Backward Compatibility

- Default behavior changes: Less verbose output (by design)
- Add `--debug` to existing scripts if verbose output is needed
- Log files can still capture everything (set file handler to DEBUG independently)
