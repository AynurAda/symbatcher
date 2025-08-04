"""
Simple demonstration of rate-limited engine with BatchScheduler.

This shows the basic usage pattern and how rate limiting works.
"""

import time
import os
import json
import logging
from datetime import datetime
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


def run_simple_demo():
    """Run a simple demonstration."""
    logger, log_file, timestamp = setup_logging()
    
    logger.info("Simple Rate-Limited Batch Processing Demo")
    logger.info("="*50)
    
    print("Simple Rate-Limited Batch Processing Demo")
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
    
    rate_limits = {
        'tokens': 50000,    # 50k tokens/min
        'requests': 500,    # 500 requests/min
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
    
    # Create test dataset - numbers 1 to 30
    dataset = [str(i) for i in range(1, 400)]
    logger.info(f"Created dataset with {len(dataset)} items: {dataset[:5]}...")
    print(f"\n✓ Created dataset with {len(dataset)} items")
    
    # Create BatchScheduler
    scheduler = BatchScheduler()
    
    # Create expression with logger
    expression_class = type('LoggingSimpleExpression', (SimpleExpression,), {
        '__init__': lambda self: SimpleExpression.__init__(self, logger)
    })
    
    # Process the batch
    logger.info("Starting batch processing...")
    logger.info("Configuration: 10 workers, batch size of 10")
    print("\nProcessing batch...")
    print("(Using 10 workers, batch size of 10)")
    
    start_time = time.time()
    
    results = scheduler.forward(
        expr=expression_class,
        num_workers=50,
        dataset=dataset,
        batch_size=50
    )
    
    duration = time.time() - start_time
    
    logger.info(f"Batch processing completed in {duration:.2f} seconds")
    
    # Show results
    print(f"\nCompleted in {duration:.2f} seconds!")
    print(f"Average: {duration/len(dataset):.3f} seconds per item")
    print(f"Throughput: {len(dataset)/duration:.1f} items/second")
    
    logger.info(f"Performance metrics:")
    logger.info(f"  - Total duration: {duration:.2f} seconds")
    logger.info(f"  - Average per item: {duration/len(dataset):.3f} seconds")
    logger.info(f"  - Throughput: {len(dataset)/duration:.1f} items/second")
    
    print("\nFirst 5 results:")
    for i in range(min(5, len(results))):
        print(f"  Number {dataset[i]}: {results[i]}")
    
    # Save results to JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "logs", f"simple_rate_limit_demo_{timestamp}_results.json")
    results_data = {
        "timestamp": timestamp,
        "configuration": {
            "num_workers": 50,
            "batch_size": 50,
            "dataset_size": len(dataset),
            "rate_limits": {
                "tokens": 50000,
                "requests": 500,
                "max_retries": 3
            }
        },
        "performance": {
            "total_duration_seconds": duration,
            "average_per_item_seconds": duration/len(dataset),
            "throughput_items_per_second": len(dataset)/duration
        },
        "results": [
            {"input": dataset[i], "output": results[i]} 
            for i in range(len(results))
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return results


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
    # Run simple demo
    run_simple_demo()
    
    # Show rate limiting options
    demo_rate_limit_options()
    
    print("\n" + "="*50)
    print("Demo completed!")


if __name__ == "__main__":
    main()