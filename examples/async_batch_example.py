"""
Example demonstrating the AsyncGPTXBatchEngine with BatchScheduler

This example shows how to use the async ChatGPT engine as a drop-in
replacement for the standard sync engine, providing better performance
for batch processing.

Note: To run this example in PyCharm:
- Right-click and select "Run 'async_batch_example'" (not pytest)
- Or run from terminal: python examples/async_batch_example.py
- PyCharm may detect test_ functions and try to run as pytest. 
  Use 'demo_error_handling' instead of 'test_error_handling' to avoid this.
"""

import time
import os
import logging
import json
from datetime import datetime
from typing import List

# Set up imports
from symai import Expression, Import, Symbol
from symai.functional import EngineRepository
from src.engines import AsyncGPTXBatchEngine
from src.func import BatchScheduler

# Configure logging
def setup_logging():
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/async_batch_example_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    
    return logger, log_filename

# Initialize logging
logger, log_file = setup_logging()

# Import the BatchScheduler
#symbatcher = Import('AynurAda/symbatcher')


class SimplePrompt(Expression):
    """A simple expression that calls the LLM with a prompt."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(self, input: str = None, **kwargs) -> str:
        topic = input
        prompt = f"Write a one-sentence interesting fact about {topic}."
        
        self.logger.debug(f"SimplePrompt.forward called with topic: '{topic}'")
        self.logger.debug(f"Prompt: '{prompt}'")
        
        try:
            # Use Symbol's query method to invoke the neurosymbolic engine
            result = Symbol(topic).query(context=topic, prompt=prompt, **kwargs)
            self.logger.debug(f"Query successful for topic '{topic}': {result}")
            return str(result)
        except Exception as e:
            self.logger.error(f"Error in SimplePrompt.forward for topic '{topic}': {str(e)}")
            raise


class ComplexAnalysis(Expression):
    """A more complex expression with context and examples."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(self, input: str = None, **kwargs) -> str:
        text = input
        # This will use the full GPTXChatEngine preparation logic
        prompt = "Analyze the sentiment of this text and explain why."
        examples = ["Happy text -> Positive sentiment", "Sad text -> Negative sentiment"]
        
        self.logger.debug(f"ComplexAnalysis.forward called with text: '{text}'")
        self.logger.debug(f"Prompt: '{prompt}'")
        self.logger.debug(f"Examples: {examples}")
        
        # Merge provided kwargs with our specific parameters
        query_kwargs = {
            'temperature': 0.7,
            **kwargs
        }
        
        try:
            result = Symbol(text).query(
                context=text,
                prompt=prompt,
                examples=examples,
                **query_kwargs
            )
            self.logger.debug(f"Analysis successful for text '{text[:50]}...': {result}")
            return str(result)
        except Exception as e:
            self.logger.error(f"Error in ComplexAnalysis.forward for text '{text[:50]}...': {str(e)}")
            raise


def compare_sync_vs_async():
    """Compare performance of sync vs async engines."""
    
    logger.info("Starting compare_sync_vs_async()")
    
    # Test data
    topics = [
        "quantum computing", "artificial intelligence", "space exploration",
        "ocean depths", "ancient history", "modern art", "climate science",
        "molecular biology", "cryptocurrency", "renewable energy"
    ]
    
    texts = [
        "I absolutely love this new feature! It's amazing!",
        "This is terrible, I'm very disappointed.",
        "The product works as expected, nothing special.",
        "Incredible innovation! This changes everything!",
        "I regret buying this, complete waste of money."
    ]
    
    # Log test data
    logger.info(f"Test topics: {topics}")
    logger.info(f"Test texts: {texts}")
    
    # Create results dictionary for logging
    results_data = {
        'topics': topics,
        'texts': texts,
        'sync_results': {},
        'async_results': {},
        'analysis_results': {}
    }
    
    print("=" * 60)
    print("Comparing Sync vs Async ChatGPT Engines")
    print("=" * 60)
    
    # Test 1: Simple prompts with sync engine (default)
    print("\n1. Testing SYNC engine with simple prompts...")
    logger.info("Test 1: Testing SYNC engine")
    scheduler = BatchScheduler()
    
    start_time = time.time()
    sync_results = scheduler.forward(
        expr=SimplePrompt,
        num_workers=5,
        dataset=topics,
        batch_size=3
    )
    sync_time = time.time() - start_time
    
    logger.info(f"Sync processing time: {sync_time:.2f} seconds")
    logger.info(f"Sync results count: {len(sync_results)}")
    
    # Log all results
    results_data['sync_results'] = {
        'time': sync_time,
        'results': list(zip(topics, sync_results))
    }
    
    print(f"Sync processing time: {sync_time:.2f} seconds")
    print(f"Results: {len(sync_results)} responses")
    for i, (topic, result) in enumerate(zip(topics[:3], sync_results[:3])):
        print(f"  - {topic}: {result}")
        logger.info(f"Sync result {i}: Topic='{topic}', Result='{result}'")
    
    # Test 2: Simple prompts with async engine
    print("\n2. Testing ASYNC engine with simple prompts...")
    logger.info("Test 2: Testing ASYNC engine")
    
    # Register the async engine
    logger.info("Registering AsyncGPTXBatchEngine")
    EngineRepository.register('neurosymbolic', AsyncGPTXBatchEngine(), allow_engine_override=True)
    
    # Create new scheduler instance to use the new engine
    scheduler = BatchScheduler()
    
    start_time = time.time()
    async_results = scheduler.forward(
        expr=SimplePrompt,
        num_workers=5,
        dataset=topics,
        batch_size=3
    )
    async_time = time.time() - start_time
    
    logger.info(f"Async processing time: {async_time:.2f} seconds")
    logger.info(f"Async results count: {len(async_results)}")
    
    # Log all results
    results_data['async_results'] = {
        'time': async_time,
        'results': list(zip(topics, async_results))
    }
    
    print(f"Async processing time: {async_time:.2f} seconds")
    print(f"Results: {len(async_results)} responses")
    for i, (topic, result) in enumerate(zip(topics[:3], async_results[:3])):
        print(f"  - {topic}: {result}")
        logger.info(f"Async result {i}: Topic='{topic}', Result='{result}'")
    
    speedup = sync_time/async_time
    print(f"\nSpeedup: {speedup:.2f}x faster with async engine!")
    logger.info(f"Performance comparison: Sync={sync_time:.2f}s, Async={async_time:.2f}s, Speedup={speedup:.2f}x")
    
    # Test 3: Complex analysis with async engine
    print("\n3. Testing ASYNC engine with complex analysis...")
    logger.info("Test 3: Testing complex analysis with ASYNC engine")
    
    scheduler = BatchScheduler()
    
    start_time = time.time()
    analysis_results = scheduler.forward(
        expr=ComplexAnalysis,
        num_workers=3,
        dataset=texts,
        batch_size=2
    )
    analysis_time = time.time() - start_time
    
    logger.info(f"Complex analysis time: {analysis_time:.2f} seconds")
    logger.info(f"Analysis results count: {len(analysis_results)}")
    
    # Log all results
    results_data['analysis_results'] = {
        'time': analysis_time,
        'results': list(zip(texts, analysis_results))
    }
    
    print(f"Complex analysis time: {analysis_time:.2f} seconds")
    print(f"Results: {len(analysis_results)} analyses")
    for i, (text, result) in enumerate(zip(texts[:2], analysis_results[:2])):
        print(f"\n  Text: '{text[:50]}...'")
        print(f"  Analysis: {result}")
        logger.info(f"Analysis result {i}: Text='{text}', Analysis='{result}'")
    
    # Save all results to JSON file
    results_json_file = log_file.replace('.log', '_results.json')
    with open(results_json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to JSON file: {results_json_file}")


def demo_error_handling():
    """Test error handling in async engine."""
    
    logger.info("Starting demo_error_handling()")
    
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Register async engine with invalid API key to test error handling
    logger.warning("Testing with invalid API key to check error handling")
    EngineRepository.register(
        'neurosymbolic', 
        AsyncGPTXBatchEngine(api_key="invalid_key"), 
        allow_engine_override=True
    )
    
    scheduler = BatchScheduler()
    
    # This should handle errors gracefully
    test_data = ["test1", "test2"]
    logger.info(f"Test data for error handling: {test_data}")
    
    results = scheduler.forward(
        expr=SimplePrompt,
        num_workers=2,
        dataset=test_data,
        batch_size=2
    )
    
    print("Results with error handling:")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: {result}")
        logger.error(f"Error handling result {i}: {result}")


def main():
    """Run the examples."""
    
    logger.info("=" * 60)
    logger.info("Starting async_batch_example.py")
    logger.info("=" * 60)
    
    # Check for API key in symbolicai config
    from symai.backend.settings import SYMAI_CONFIG
    
    api_key = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_API_KEY')
    if not api_key:
        error_msg = "Error: Please configure your API key in symbolicai"
        logger.error(error_msg)
        print(error_msg)
        print("The API key should be set in your symai.config.json file")
        print(f"Configuration file: {os.path.expanduser('~')}/.symai/symai.config.json")
        print("Add or update: \"NEUROSYMBOLIC_ENGINE_API_KEY\": \"your-openai-api-key\"")
        return
    
    # Log API key info (mask most of it for security)
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    logger.info(f"API key configured: {masked_key}")
    logger.info(f"Model: {SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL', 'Not set')}")
    
    # Run comparison
    compare_sync_vs_async()
    
    # Uncomment to test error handling
    # demo_error_handling()
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("Example completed successfully")
    logger.info(f"Log file saved to: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()