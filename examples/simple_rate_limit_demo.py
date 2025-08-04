"""
Simple demonstration of rate-limited engine with BatchScheduler.

This shows the basic usage pattern and how rate limiting works.
"""

import time
import os
import sys
import json
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symai import Expression, Symbol
from symai.functional import EngineRepository
from src.engines.async_chatgpt_rate_limited import AsyncGPTXBatchEngine
from src.func import BatchScheduler


class SimpleExpression(Expression):
    """A simple expression that asks for a number fact."""
    
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
    
    def forward(self, input: str = None, **kwargs) -> str:
        number = input
        prompt = f"Tell me one interesting fact about the number {number} in 10 words or less."
        
        if self.logger:
            self.logger.info(f"Processing number: {number}")
            self.logger.info(f"Prompt: {prompt}")
        
        result = Symbol(number).query(
            context=number,
            prompt=prompt,
            temperature=0.5,
            max_completion_tokens=30,
            **kwargs
        )
        
        if self.logger:
            self.logger.info(f"Response for {number}: {str(result)}")
        
        return str(result)


def setup_logging():
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use absolute path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"simple_rate_limit_demo_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('simple_rate_limit_demo')
    logger.info(f"Logging started. Log file: {log_file}")
    
    return logger, log_file, timestamp


def benchmark_batch_sizes():
    """Benchmark different batch sizes with rate-limited engine."""
    logger, log_file, timestamp = setup_logging()
    
    logger.info("Rate-Limited Batch Processing Benchmark")
    logger.info("="*50)
    
    print("Rate-Limited Batch Processing Benchmark")
    print("="*50)
    
    # Check for API key
    from symai.backend.settings import SYMAI_CONFIG
    api_key = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_API_KEY')
    if not api_key:
        logger.error("API key not configured in symbolicai")
        logger.error("Please set NEUROSYMBOLIC_ENGINE_API_KEY in symai.config.json")
        print("\nERROR: API key not configured in symbolicai")
        print("Please set NEUROSYMBOLIC_ENGINE_API_KEY in symai.config.json")
        return
    
    # Create rate-limited engine with moderate limits
    logger.info("Creating rate-limited engine...")
    print("\nCreating rate-limited engine...")
    
    # Using custom limits slightly below Tier 2 for demonstration
    # (Tier 1: 40K TPM/500 RPM for GPT-3.5, Tier 2: 80K TPM/5000 RPM)
    rate_limits = {
        'tokens': 36000,    # 90% of Tier 1 limit (40k tokens/min)
        'requests': 450,    # 90% of Tier 1 limit (500 requests/min)
        'max_retries': 3
    }
    
    logger.info(f"Rate limits configuration: {json.dumps(rate_limits, indent=2)}")
    
    engine = AsyncGPTXBatchEngine(
        model='gpt-3.5-turbo',
        rate_limits=rate_limits
    )
    
    # Register the engine
    EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
    logger.info("Rate-limited engine registered successfully")
    print("✓ Rate-limited engine registered")
    
    # Create larger test dataset for benchmarking
    dataset = [str(i) for i in range(1, 101)]  # 100 items for testing
    logger.info(f"Created dataset with {len(dataset)} items")
    print(f"\n✓ Created dataset with {len(dataset)} items")
    
    # Create expression with logger
    expression_class = type('LoggingSimpleExpression', (SimpleExpression,), {
        '__init__': lambda self: SimpleExpression.__init__(self, logger)
    })
    
    # Test different batch sizes
    batch_sizes = [1, 10, 20, 50, 100]
    benchmark_results = []
    
    logger.info("Testing different batch sizes...")
    print("\nTesting different batch sizes...")
    print("="*60)
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        logger.info(f"Testing batch size: {batch_size}")
        
        # Use subset of data for each batch size
        test_data = dataset[:batch_size]
        
        # Create new scheduler for each test
        scheduler = BatchScheduler()
        
        # num_workers should equal batch_size
        num_workers = batch_size
        
        logger.info(f"Using {num_workers} workers for batch size {batch_size} (num_workers = batch_size)")
        
        start_time = time.time()
        
        results = scheduler.forward(
            expr=expression_class,
            num_workers=num_workers,
            dataset=test_data,
            batch_size=batch_size
        )
        
        duration = time.time() - start_time
        
        # Calculate metrics
        requests_per_second = len(test_data) / duration
        avg_time_per_request = duration / len(test_data)
        
        # Store results
        batch_result = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'total_requests': len(test_data),
            'total_time': duration,
            'requests_per_second': requests_per_second,
            'avg_time_per_request': avg_time_per_request,
            'sample_results': [(test_data[i], results[i]) for i in range(min(3, len(results)))]
        }
        benchmark_results.append(batch_result)
        
        logger.info(f"Batch size {batch_size} - Total time: {duration:.2f}s")
        logger.info(f"Batch size {batch_size} - Requests/second: {requests_per_second:.2f}")
        logger.info(f"Batch size {batch_size} - Avg time/request: {avg_time_per_request:.3f}s")
        
        print(f"  Total time: {duration:.2f} seconds")
        print(f"  Requests processed: {len(results)}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Average time per request: {avg_time_per_request:.3f} seconds")
        print(f"  Sample result: Number {test_data[0]} -> {results[0]}")
    
    # Generate summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (Rate-Limited)")
    print("="*60)
    
    print(f"{'Batch Size':<12} {'Total Time':<15} {'Requests/sec':<15} {'Avg Time/Req':<15}")
    print("-" * 60)
    
    for result in benchmark_results:
        print(f"{result['batch_size']:<12} {result['total_time']:.2f}s{'':<9} {result['requests_per_second']:.2f}{'':<10} {result['avg_time_per_request']:.3f}s")
    
    # Find best and worst performers
    best_batch = max(benchmark_results, key=lambda x: x['requests_per_second'])
    worst_batch = min(benchmark_results, key=lambda x: x['requests_per_second'])
    
    performance_gain = best_batch['requests_per_second'] / worst_batch['requests_per_second']
    
    print(f"\nBest performance: Batch size {best_batch['batch_size']} with {best_batch['requests_per_second']:.2f} requests/sec")
    print(f"Worst performance: Batch size {worst_batch['batch_size']} with {worst_batch['requests_per_second']:.2f} requests/sec")
    print(f"Performance gain: {performance_gain:.2f}x")
    
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    for result in benchmark_results:
        logger.info(f"Batch {result['batch_size']}: {result['total_time']:.2f}s, {result['requests_per_second']:.2f} req/s, {result['avg_time_per_request']:.3f}s/req")
    logger.info(f"Best: Batch {best_batch['batch_size']} ({best_batch['requests_per_second']:.2f} req/s)")
    logger.info(f"Performance gain: {performance_gain:.2f}x")
    
    # Save benchmark results to JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "logs", f"simple_rate_limit_demo_{timestamp}_results.json")
    results_data = {
        "test_type": "rate_limited_batch_size_benchmark",
        "timestamp": timestamp,
        "rate_limits": {
            "tokens": 50000,
            "requests": 500,
            "max_retries": 3
        },
        "batch_tests": benchmark_results,
        "summary": {
            "best_batch_size": best_batch['batch_size'],
            "best_requests_per_second": best_batch['requests_per_second'],
            "worst_batch_size": worst_batch['batch_size'],
            "worst_requests_per_second": worst_batch['requests_per_second'],
            "performance_gain": performance_gain
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return benchmark_results


def demo_rate_limit_options():
    """Show different rate limiting options available."""
    print("\n" + "="*50)
    print("Rate Limiting Options:")
    print("="*50)
    
    print("\n1. No rate limiting (rate_limits=None):")
    print("   engine = AsyncGPTXBatchEngine(model='gpt-3.5-turbo', rate_limits=None)")
    print("   → Disables rate limiting completely (use with caution!)")
    
    print("\n2. Default rate limits (rate_limits='default'):")
    print("   engine = AsyncGPTXBatchEngine(model='gpt-3.5-turbo', rate_limits='default')")
    print("   → Uses safe defaults based on the model")
    
    print("\n3. Custom rate limits (rate_limits={...}):")
    print("   engine = AsyncGPTXBatchEngine(model='gpt-3.5-turbo', rate_limits={")
    print("       'tokens': 50000,")
    print("       'requests': 500,")
    print("       'max_retries': 3")
    print("   })")
    print("   → Fine-grained control over rate limiting")


def main():
    """Run the demonstration."""
    # Run benchmark
    benchmark_batch_sizes()
    
    # Show rate limiting options
    demo_rate_limit_options()
    
    print("\n" + "="*50)
    print("Benchmark completed!")


if __name__ == "__main__":
    main()