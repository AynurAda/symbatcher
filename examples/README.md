# Async ChatGPT Engine Examples

This directory contains examples demonstrating the AsyncGPTXBatchEngine.

## Setup

1. Install required dependencies:
```bash
pip install openai symbolicai
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Running the Example

```bash
python async_batch_example.py
```

## What the Example Demonstrates

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