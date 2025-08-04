# Rate Limiting Examples

This directory contains examples demonstrating how to use the rate-limited async ChatGPT engine with BatchScheduler.

## Important Notes

1. **Always use BatchScheduler**: The async engines (both rate-limited and standard) are designed to work with BatchScheduler. Do not use `Symbol.query()` directly with these engines.

2. **Create BatchScheduler after engine registration**: Always create a new BatchScheduler instance after registering an engine with `EngineRepository.register()`.

## Examples

### 1. simple_rate_limit_demo.py

A simple demonstration of rate limiting with BatchScheduler:
- Shows basic usage with 30 number facts
- Optional comparison between rate-limited and non-rate-limited engines
- Clear performance metrics

```bash
python simple_rate_limit_demo.py
```

### 2. Additional Examples (Coming Soon)

More comprehensive examples are being developed to demonstrate:
- Large-scale batch processing with rate limiting
- Custom rate limit strategies
- Integration with different OpenAI models
- Advanced retry and backoff configurations

## Configuration

### Environment Variables
```bash
# Optional - uses model defaults if not set
export SYMBATCHER_OPENAI_TOKENS_PER_MIN=50000
export SYMBATCHER_OPENAI_REQUESTS_PER_MIN=2000
export SYMBATCHER_OPENAI_MAX_RETRIES=5

# Disable rate limiting
export SYMBATCHER_RATE_LIMITING_ENABLED=false
```

### Programmatic Configuration
```python
# Use model defaults
engine = AsyncGPTXBatchEngine(model='gpt-4')

# Custom rate limits
engine = AsyncGPTXBatchEngine(
    model='gpt-4',
    rate_limits={
        'tokens': 50000,
        'requests': 2000,
        'max_retries': 5
    }
)

# No rate limiting
engine = AsyncGPTXBatchEngine(model='gpt-4', rate_limits=None)
```

## Common Issues

### executor_callback Error
If you see errors like `'AsyncGPTXBatchEngine' object has no attribute 'executor_callback'`, this means:
- You're trying to use `Symbol.query()` directly instead of BatchScheduler
- Solution: Always use BatchScheduler with these engines

### API Key Configuration
The examples expect your OpenAI API key to be configured in symbolicai:
- Configuration file: `~/.symai/symai.config.json`
- Add: `"NEUROSYMBOLIC_ENGINE_API_KEY": "your-openai-api-key"`

## Best Practices

1. **Always use BatchScheduler** for batch processing
2. **Set conservative rate limits** to avoid hitting API limits
3. **Monitor throughput** to ensure rate limiting is working
4. **Use appropriate batch sizes** based on your rate limits
5. **Create new BatchScheduler** after changing engines