# symbatcher

## Description
Symbatcher is a tool that leverages the symbolicai library to provide efficient batch scheduling capabilities.
Currently only 1 engine at a time is supported. Engine switching to be introduced in a future release.

## Installation

To use symbatcher, you first need to install the symbolicai library. You can do this using pip:

```bash
pip install symbolicai
```

## Usage

### Simple Usage (Recommended)

```python
from symai import Import, Expression, Symbol

# Define your Expression class
class TestExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        return Symbol(input).query("Summarize this input", **kwargs).value

# Import and use BatchScheduler
BatchScheduler = Import('AynurAda/symbatcher')
scheduler = BatchScheduler()  # Automatically manages its own event loop

# Run batch processing
results = scheduler(
    expr=TestExpression,           # Pass the Expression class (not an instance)
    num_workers=2,                 # Number of parallel workers
    dataset=["test1", "test2", "test3"],  # List of inputs to process
    batch_size=2                   # Optional: batch size (default=5)
)

# Clean up when done
scheduler.cleanup()
```

### Using Context Manager (Auto-cleanup)

```python
# Use with statement for automatic cleanup
with BatchScheduler() as scheduler:
    results = scheduler(
        expr=TestExpression,
        num_workers=2,
        dataset=["test1", "test2", "test3"],
        batch_size=2
    )
    # Cleanup happens automatically when exiting the context
```

### Advanced Usage (Custom Event Loop)

For integration with existing async applications, you can provide your own event loop:

```python
import asyncio
import threading

# Create your own event loop
main_loop = asyncio.new_event_loop()
thread = threading.Thread(
    target=BatchScheduler.start_loop,  # Use the static method
    args=(main_loop,), 
    daemon=True
)
thread.start()

# Create scheduler with your event loop
scheduler = BatchScheduler(main_loop=main_loop)

# ... use the scheduler ...

# Clean up your custom event loop
BatchScheduler.close_loop(main_loop, thread)
```

See `example_usage.py` for complete working examples including multiple schedulers and advanced patterns.



 
