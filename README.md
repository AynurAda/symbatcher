# symbatcher

## Description
Symbatcher is a high-performance batch scheduling library for the symbolicai framework. It provides:
- Efficient parallel batch processing of AI/ML expressions
- Async engine support for improved throughput (2-5x faster)
- Built-in rate limiting for API compliance
- Thread-safe concurrent execution
- Queue-based work distribution

Currently only 1 engine at a time is supported. Engine switching to be introduced in a future release.

## Installation

To use symbatcher, you first need to install the symbolicai library. You can do this using pip:

```bash
pip install symbolicai
```

## Important: Expression Requirements

### Input Parameter Requirement

**All Expression classes used with BatchScheduler MUST accept an `input` parameter in their `forward()` method.**

The BatchScheduler passes each item from your dataset as the `input` parameter to the expression's forward method:

```python
def forward(self, input, **kwargs):
    # Your processing logic here
```

⚠️ **Common Error**: If your expression's forward method doesn't accept an `input` parameter, you'll get:
```
TypeError: forward() got an unexpected keyword argument 'input'
```

### Handling Multiple Parameters

If you need to pass multiple values to your expression, use a dictionary as the input:

```python
# Define an expression that expects multiple values
class MultiParamExpression(Expression):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, **kwargs):
        # Extract multiple values from the input dictionary
        text = input['text']
        language = input['language']
        max_length = input.get('max_length', 100)  # with default
        
        # Process using the multiple parameters
        prompt = f"Translate this {language} text: {text}"
        return Symbol(text).query(prompt, **kwargs).value

# Use with BatchScheduler
dataset = [
    {'text': 'Hello world', 'language': 'English', 'max_length': 50},
    {'text': 'Bonjour monde', 'language': 'French'},
    {'text': 'Hola mundo', 'language': 'Spanish', 'max_length': 75}
]

scheduler = BatchScheduler()
results = scheduler(
    expr=MultiParamExpression,
    num_workers=3,
    dataset=dataset,  # List of dictionaries
    batch_size=2
)
```

Additional parameters can still be passed via `**kwargs` which will be forwarded to all expressions:

```python
results = scheduler(
    expr=TestExpression,
    num_workers=2,
    dataset=["input1", "input2"],
    temperature=0.7,  # This will be in kwargs
    max_tokens=100    # This will also be in kwargs
)
```

### Common Input Patterns

```python
# Simple string inputs
dataset = ["text1", "text2", "text3"]

# Dictionary inputs for multiple parameters
dataset = [
    {"question": "What is AI?", "context": "Article about AI..."},
    {"question": "How does ML work?", "context": "ML tutorial..."}
]

# Mixed data types
dataset = [
    {"image_path": "/path/to/image1.jpg", "style": "artistic"},
    {"image_path": "/path/to/image2.jpg", "style": "realistic"}
]

# With metadata
dataset = [
    {"id": 1, "text": "Process this", "priority": "high"},
    {"id": 2, "text": "Process that", "priority": "low"}
]
```

## Async Engines (New!)

Symbatcher now includes async engine implementations for improved performance:

### AsyncGPTXBatchEngine
A drop-in replacement for GPTXChatEngine that processes requests concurrently:
- 2-5x faster for batch processing
- No code changes required - just register and use
- Full compatibility with all GPTXChatEngine features

### Rate-Limited Async Engine
Includes built-in rate limiting to prevent API throttling:
- Configurable rate limits (tokens/requests per minute)
- Automatic retry with exponential backoff
- Environment variable configuration support

### Performance Benchmarks

#### Rate-Limited Engine Performance (Tier 1: 36k tokens/min, 450 requests/min)
| Batch Size | Requests/sec | Avg Time/Request | Performance |
|------------|--------------|------------------|-------------|
| 1          | 0.72         | 1.381s          | Baseline    |
| 10         | 7.44         | 0.134s          | 10.3x faster |
| 20         | 3.94         | 0.254s          | 5.4x faster |
| 50         | **18.12**    | **0.055s**      | **25x faster** |
| 100        | 1.70         | 0.589s          | 2.3x faster |

**Key findings:**
- Optimal batch size: 50 (achieves 18.12 requests/sec)
- Performance gain: Up to 25x faster than single requests
- Batch size 100 shows degraded performance due to rate limit constraints
- Sweet spot for rate-limited processing: batch sizes between 10-50

**Note:** Default limits are set to OpenAI Tier 1 ($5 paid users). Higher tier users can increase limits via environment variables or custom configuration.

```python
from symai import EngineRepository
from src.engines.async_chatgpt_rate_limited import AsyncGPTXBatchEngine

# Use with rate limiting (recommended)
engine = AsyncGPTXBatchEngine(model='gpt-4', rate_limits='default')
EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)

# Or without rate limiting for maximum speed
engine = AsyncGPTXBatchEngine(model='gpt-4', rate_limits=None)
EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
```

See `examples/` directory for complete examples.

## Usage

### Simple Usage (Recommended)

```python
from symai import Import, Expression, Symbol

# Define your Expression class
class TestExpression(Expression):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, **kwargs):
        # 'input' parameter is REQUIRED - it receives each item from the dataset
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

## Examples

The `examples/` directory contains several demonstration scripts:

- **example_usage.py** - Basic BatchScheduler usage patterns
- **async_batch_example.py** - Async engine performance comparison
- **simple_rate_limit_demo.py** - Rate limiting demonstration
- **RATE_LIMITING_EXAMPLES.md** - Comprehensive rate limiting guide

## Testing

Run the test suite with pytest:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_batchscheduler.py

# Run async engine tests
pytest tests/test_async_chatgpt_rate_limited.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
