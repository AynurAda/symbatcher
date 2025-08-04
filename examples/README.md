# Symbatcher Examples

This directory contains examples demonstrating various features of Symbatcher, including async engines and rate limiting.

## Setup

1. Install required dependencies:
```bash
pip install openai symbolicai
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Available Examples

### 1. example_usage.py
Basic BatchScheduler usage patterns including:
- Simple batch processing
- Context manager usage
- Custom event loop management
- Multiple scheduler instances

```bash
python example_usage.py
```

### 2. async_batch_example.py
Demonstrates the AsyncGPTXBatchEngine performance improvements:
- Sync vs async engine comparison
- Drop-in replacement usage
- Performance metrics

```bash
python async_batch_example.py
```

### 3. simple_rate_limit_demo.py
Shows rate limiting in action:
- Configurable rate limits
- Automatic retry on throttling
- Throughput monitoring

```bash
python simple_rate_limit_demo.py
```

## What the Examples Demonstrate

1. **Performance Comparison**: Shows sync vs async engine performance
2. **Drop-in Replacement**: Demonstrates how to replace the sync engine without changing any other code
3. **Complex Expressions**: Shows that all GPTXChatEngine features work (prompts, examples, temperature, etc.)
4. **Error Handling**: Demonstrates graceful error handling in batch processing

## Key Benefits

- **Concurrent API Calls**: The async engine processes multiple requests in parallel
- **Better Throughput**: Typically 2-5x faster for batch processing
- **No Code Changes**: Just register the async engine - everything else works the same
- **Full Compatibility**: All features of GPTXChatEngine are preserved

## How It Works

```python
# Register the async engine (one line change!)
EngineRepository.register('neurosymbolic', AsyncGPTXBatchEngine(), allow_engine_override=True)

# Use BatchScheduler exactly as before
scheduler = BatchScheduler()
results = scheduler.forward(expr=expr, num_workers=5, dataset=data, batch_size=3)
```

The async engine handles all the complexity internally while providing the same synchronous interface that BatchScheduler expects.

## Configuration Options

### Environment Variables
```bash
# Rate limiting configuration
export SYMBATCHER_OPENAI_TOKENS_PER_MIN=50000
export SYMBATCHER_OPENAI_REQUESTS_PER_MIN=2000
export SYMBATCHER_OPENAI_MAX_RETRIES=5
export SYMBATCHER_RATE_LIMITING_ENABLED=true

# API key configuration (symbolicai)
# Add to ~/.symai/symai.config.json:
# "NEUROSYMBOLIC_ENGINE_API_KEY": "your-openai-api-key"
```

### Programmatic Configuration
See `simple_rate_limit_demo.py` for examples of programmatic rate limit configuration.

## Logging

Examples create detailed logs in the `logs/` directory:
- Execution logs with timestamps
- Results in JSON format
- Performance metrics

## Troubleshooting

1. **Import Errors**: Make sure you're in the project root when running examples
2. **API Key Issues**: Check your symbolicai configuration file
3. **Rate Limit Errors**: Reduce batch_size or num_workers parameters
4. **executor_callback Errors**: Always use BatchScheduler with async engines