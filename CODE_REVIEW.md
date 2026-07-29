# Code Review & Optimization Report - Veo Project

## Executive Summary

The codebase is well-structured and functional, but has several optimization opportunities that can improve performance, maintainability, and reliability. Key issues identified:

1. **Performance**: Inefficient video I/O patterns (repeated opens/closes)
2. **Code Quality**: Inconsistent type hints, redundant conversions
3. **Maintainability**: Hardcoded values, inconsistent error handling
4. **Architecture**: Opportunity for better resource management

---

## Detailed Findings

### 1. **detect_track.py** - Critical Performance Issues

**Issue**: `_fragment_features()` opens/closes video for each fragment sample
- **Impact**: For a 90-minute match at 25fps, this could mean thousands of unnecessary I/O operations
- **Recommendation**: Keep video open for the entire feature extraction pass

**Issue**: Repeated numpy operations and type conversions
- `float(row.px)`, `int(np.clip(...))` called repeatedly
- **Impact**: Small but cumulative overhead in hot loops

**Optimizations Applied**:
- Cache video capture handle across fragment features
- Pre-allocate numpy arrays where possible
- Use vectorized operations for distance calculations
- Reduce redundant type conversions

---

### 2. **team_assignment.py** - Video I/O Inefficiency

**Issue**: Opens video in loop for each track sample
- Same as detect_track, but in a different function
- Could be batched into single pass

**Optimizations Applied**:
- Pre-read all required frames once
- Batch color extraction

---

### 3. **main.py** - Structure & Clarity

**Issues**:
- Late import of `cv2` (line 127) - should be at top
- Long argument list to `report.build_pdf()`
- No error handling for file not found cases

**Optimizations Applied**:
- Move imports to top
- Add input validation early
- Cleaner argument passing with dataclass or dict

---

### 4. **stats.py** - Minor Improvements

**Issues**:
- `_smooth()` creates unnecessary copies with `.copy()`
- Redundant filtering in `possession_proxy()`

**Optimizations Applied**:
- Reduce unnecessary copies
- Vectorize where possible

---

### 5. **player_rating.py** - Code Style

**Issues**:
- Lambda functions could be more explicit
- Good overall quality but some repetitive dict access

**Optimizations Applied**:
- Extract named functions from lambdas
- Use `.get()` more consistently

---

### 6. **General Issues Across Files**

**Logging & Error Handling**:
- Inconsistent use of print vs logging
- No graceful handling of corrupted video frames
- Silent failures in some edge cases

**Type Hints**:
- Inconsistent use of `|` vs `Union` (mixing old and new syntax)
- Some functions lack return type hints

**Constants**:
- Magic numbers scattered throughout (e.g., 0.25, 3.6, 20.0)
- Should be consolidated in constants module

---

## Performance Impact Estimates

| Optimization | Estimated Impact | Effort |
|---|---|---|
| Video I/O batching (detect_track) | **20-40% faster** | Medium |
| Remove redundant type conversions | **5-10% faster** | Low |
| Vectorize distance calcs | **2-5% faster** | Low |
| Video I/O batching (team_assign) | **15-25% faster** | Low |
| **Total Combined** | **~40-50% faster** | Medium |

---

## Recommendations Priority

### P0 (Critical for Performance)
- [ ] Optimize video I/O in `detect_track.py`
- [ ] Optimize video I/O in `team_assignment.py`

### P1 (Important for Quality)
- [ ] Consolidate constants
- [ ] Add proper error handling for edge cases
- [ ] Improve logging vs print statements

### P2 (Nice to Have)
- [ ] Type hints consistency
- [ ] Extract magic numbers to named constants
- [ ] Refactor lambdas to named functions

---

## Files Modified

1. `src/detect_track.py` - Major optimizations
2. `src/team_assignment.py` - Video I/O optimization
3. `src/stats.py` - Minor efficiency improvements
4. `src/player_rating.py` - Code clarity
5. `src/constants.py` - NEW: centralized constants
6. `main.py` - Structure improvements

