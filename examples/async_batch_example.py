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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symai import Expression, Import, Symbol
from symai.functional import EngineRepository
from src.engines import AsyncGPTXBatchEngine
from src.func import BatchScheduler

# Configure logging
def setup_logging():
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(logs_dir, f'async_batch_example_{timestamp}.log')
    
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


def benchmark_batch_sizes():
    """Benchmark non-rate-limited async engine with different batch sizes."""
    
    logger.info("Starting benchmark_batch_sizes()")
    
    # Create more test data for larger batch sizes
    base_topics = [
        "quantum computing", "artificial intelligence", "space exploration",
        "ocean depths", "ancient history", "modern art", "climate science",
        "molecular biology", "cryptocurrency", "renewable energy",
        "machine learning", "robotics", "nanotechnology", "genetics",
        "astronomy", "psychology", "economics", "philosophy", "mathematics",
        "physics", "chemistry", "biology", "geology", "meteorology",
        "archaeology", "anthropology", "sociology", "linguistics", "neuroscience",
        "medicine", "engineering", "architecture", "music theory", "literature",
        "history", "geography", "political science", "law", "education",
        "computer science", "data science", "cybersecurity", "blockchain", "virtual reality",
        "augmented reality", "biotechnology", "renewable materials", "sustainability", "ecology",
        "marine biology", "astrophysics", "quantum mechanics", "relativity theory", "string theory",
        "cosmology", "particle physics", "nuclear physics", "thermodynamics", "fluid dynamics",
        "electromagnetism", "optics", "acoustics", "materials science", "metallurgy",
        "ceramics", "polymers", "composites", "semiconductors", "superconductors",
        "photonics", "plasmonics", "metamaterials", "biomaterials", "nanomaterials",
        "energy storage", "fuel cells", "solar energy", "wind energy", "hydroelectric power",
        "geothermal energy", "nuclear energy", "fusion energy", "carbon capture", "climate modeling",
        "weather prediction", "seismology", "volcanology", "oceanography", "hydrology",
        "glaciology", "paleontology", "evolutionary biology", "ecology", "conservation biology",
        "synthetic biology", "bioinformatics", "proteomics", "genomics", "metabolomics"
    ]
    
    # Ensure we have at least 100 topics
    topics = base_topics[:100]
    
    # Log test data
    logger.info(f"Test topics count: {len(topics)}")
    logger.info(f"First 5 topics: {topics[:5]}")
    
    # Create results dictionary for logging
    results_data = {
        'test_type': 'batch_size_performance',
        'topics_count': len(topics),
        'batch_tests': [],
        'summary': {}
    }
    
    print("=" * 60)
    print("Testing AsyncGPTXBatchEngine with Different Batch Sizes")
    print("=" * 60)
    
    # Register the async engine (non-rate-limited)
    logger.info("Registering AsyncGPTXBatchEngine (non-rate-limited)")
    EngineRepository.register('neurosymbolic', AsyncGPTXBatchEngine(), allow_engine_override=True)
    
    # Test different batch sizes
    batch_sizes = [10, 20, 50, 100]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        logger.info(f"Testing batch size: {batch_size}")
    
        # Use the appropriate subset of topics
        test_topics = topics[:batch_size]
        
        # Create new scheduler instance
        scheduler = BatchScheduler()
        
        # Set num_workers equal to batch_size for optimal performance
        num_workers = batch_size
        
        logger.info(f"Using {batch_size} workers for batch size {batch_size} (num_workers = batch_size)")
        
        start_time = time.time()
        results = scheduler.forward(
            expr=SimplePrompt,
            num_workers=batch_size,
            dataset=test_topics,
            batch_size=batch_size
        )
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        requests_per_second = len(test_topics) / elapsed_time
        avg_time_per_request = elapsed_time / len(test_topics)
        
        # Log detailed results
        batch_result = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'total_requests': len(test_topics),
            'total_time': elapsed_time,
            'requests_per_second': requests_per_second,
            'avg_time_per_request': avg_time_per_request,
            'sample_results': list(zip(test_topics[:3], results[:3]))
        }
        results_data['batch_tests'].append(batch_result)
        
        logger.info(f"Batch size {batch_size} - Total time: {elapsed_time:.2f}s")
        logger.info(f"Batch size {batch_size} - Requests/second: {requests_per_second:.2f}")
        logger.info(f"Batch size {batch_size} - Avg time/request: {avg_time_per_request:.3f}s")
        
        print(f"  Total time: {elapsed_time:.2f} seconds")
        print(f"  Requests processed: {len(results)}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Average time per request: {avg_time_per_request:.3f} seconds")
        print(f"  Sample result: {test_topics[0]} -> {results[0]}")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    summary_table = []
    for test in results_data['batch_tests']:
        summary_table.append({
            'Batch Size': test['batch_size'],
            'Total Time (s)': f"{test['total_time']:.2f}",
            'Requests/sec': f"{test['requests_per_second']:.2f}",
            'Avg Time/Request (s)': f"{test['avg_time_per_request']:.3f}"
        })
    
    # Print formatted table
    print(f"{'Batch Size':<12} {'Total Time':<15} {'Requests/sec':<15} {'Avg Time/Req':<15}")
    print("-" * 60)
    for row in summary_table:
        print(f"{row['Batch Size']:<12} {row['Total Time (s)']:<15} {row['Requests/sec']:<15} {row['Avg Time/Request (s)']:<15}")
    
    # Find optimal batch size
    best_batch = max(results_data['batch_tests'], key=lambda x: x['requests_per_second'])
    worst_batch = min(results_data['batch_tests'], key=lambda x: x['requests_per_second'])
    
    results_data['summary'] = {
        'best_batch_size': best_batch['batch_size'],
        'best_requests_per_second': best_batch['requests_per_second'],
        'worst_batch_size': worst_batch['batch_size'],
        'worst_requests_per_second': worst_batch['requests_per_second'],
        'performance_gain': best_batch['requests_per_second'] / worst_batch['requests_per_second']
    }
    
    print(f"\nBest performance: Batch size {best_batch['batch_size']} with {best_batch['requests_per_second']:.2f} requests/sec")
    print(f"Worst performance: Batch size {worst_batch['batch_size']} with {worst_batch['requests_per_second']:.2f} requests/sec")
    print(f"Performance gain: {results_data['summary']['performance_gain']:.2f}x")
    
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    for row in summary_table:
        logger.info(f"Batch {row['Batch Size']}: {row['Total Time (s)']}s, {row['Requests/sec']} req/s, {row['Avg Time/Request (s)']}s/req")
    logger.info(f"Best: Batch {best_batch['batch_size']} ({best_batch['requests_per_second']:.2f} req/s)")
    logger.info(f"Performance gain: {results_data['summary']['performance_gain']:.2f}x")
    
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
    logger.info("Starting async_batch_example.py - Batch Size Performance Test")
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
    
    # Run batch size benchmarks
    benchmark_batch_sizes()
    
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