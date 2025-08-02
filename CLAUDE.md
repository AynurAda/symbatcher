# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Symbatcher is a Python library that provides efficient batch scheduling capabilities for the symbolicai framework. It enables parallel execution of AI/ML expressions with optimized batch processing.

## Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_batchscheduler.py

# Run with timeout (tests have built-in timeouts)
pytest -v

# Run a specific test function
pytest tests/test_batchscheduler.py::test_simple_batch
```

### Common Development Tasks
- **Install for development**: `pip install -e .` (Note: No setup.py exists, so install dependencies manually)
- **Run specific test scenarios**:
  - Basic functionality: `pytest tests/test_batchscheduler.py`
  - Error handling: `pytest tests/test_batchscheduler_error.py`
  - Performance comparison: `pytest tests/test_compare_batch_and_single.py`

## Architecture

### Core Components

1. **BatchScheduler (src/func.py)**
   - Main class extending symbolicai's Expression
   - Manages concurrent execution using ThreadPoolExecutor
   - Implements queue-based batch processing with configurable batch sizes
   - Handles executor callbacks bound directly to the engine

2. **Concurrency Model**
   - Uses ThreadPoolExecutor for parallel processing
   - Queue-based system for managing work items
   - Thread-safe operations with proper locking
   - Configurable number of worker threads

3. **Integration Points**
   - Extends symbolicai.Expression base class
   - Uses EngineRepository for engine management
   - Direct executor_callback binding to engine for response handling
   - Supports nested expressions with dependency resolution

### Key Design Patterns

- **Expression Pattern**: All batch operations inherit from symbolicai's Expression class
- **Queue-Based Processing**: Work items are queued and processed in batches
- **Callback System**: Results are handled through executor callbacks bound to the engine
- **Mock Testing**: Comprehensive mock engine system for unit testing

### Important Implementation Details

- The library currently supports only one engine at a time
- Batch size and number of workers are configurable via forward() method parameters
- Error handling includes graceful degradation and comprehensive exception catching
- All expressions must implement a forward() method that accepts input and kwargs
- The module is imported via symbolicai's Import system: `Import('AynurAda/symbatcher')`

## Testing Strategy

Tests use mock engines to simulate various scenarios:
- Normal operation with configurable delays
- Error conditions and exception handling
- Nested expression execution
- Performance comparisons between batch and single processing

Mock engines support:
- Controlled response delays
- Error simulation
- Call tracking for verification

## Reference Projects

The `reference_projects/` folder contains source code from related projects for reference:

### symbolicai (ExtensityAI)
- Location: `reference_projects/symbolicai/`
- Purpose: Core symbolicai framework that this project extends
- Key areas to reference:
  - Expression base class implementation
  - Engine and EngineRepository patterns
  - Import system for loading external modules
  - Callback and execution patterns