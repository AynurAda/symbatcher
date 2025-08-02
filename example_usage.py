import asyncio
import threading
import time
from symai import Import, Expression, Symbol
from src.func import BatchScheduler


# ------------------------------------
# Example Expression Classes
# ------------------------------------
class RiskManager(Expression):
    """Example expression for risk assessment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        return Symbol(input).query("Assess the risk level of this scenario", **kwargs).value


class DataProcessor(Expression):
    """Example expression for data processing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        return Symbol(input).query("Process and summarize this data", **kwargs).value


class SimpleAnalyzer(Expression):
    """Example expression for simple analysis."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        return Symbol(input).query("Analyze this input and provide insights", **kwargs).value


# ------------------------------------
# Example 1: Simple Usage (Recommended)
# ------------------------------------
def simple_usage_example():
    """
    Simplest way to use BatchScheduler - it manages its own event loop.
    """
    print("=== Simple Usage Example ===")
    
    # Create scheduler - it will manage its own event loop
    scheduler = BatchScheduler()
    
    # Example dataset
    risk_inputs = [
        "High volatility stock trading",
        "Investment in emerging markets",
        "Cryptocurrency portfolio"
    ]
    
    start_time = time.time()
    
    # Run batch processing
    results = scheduler.forward(
        RiskManager,
        num_workers=2,
        batch_size=2,
        dataset=risk_inputs
    )
    
    print(f"Risk assessment completed in {time.time() - start_time:.2f} seconds")
    for i, result in enumerate(results):
        print(f"  Risk {i+1}: {result}")
    
    # Clean up when done
    scheduler.cleanup()
    print("Cleanup completed\n")


# ------------------------------------
# Example 2: Context Manager Usage
# ------------------------------------
def context_manager_example():
    """
    Using BatchScheduler as a context manager for automatic cleanup.
    """
    print("=== Context Manager Example ===")
    
    data_inputs = [
        "Q1 sales report: $1.2M revenue, 15% growth",
        "Customer feedback: 89% satisfaction rate",
        "Market analysis: 3% market share increase"
    ]
    
    # Use context manager for automatic cleanup
    with BatchScheduler() as scheduler:
        start_time = time.time()
        
        results = scheduler.forward(
            DataProcessor,
            num_workers=2,
            batch_size=3,
            dataset=data_inputs
        )
        
        print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
        for i, result in enumerate(results):
            print(f"  Processed {i+1}: {result}")
    
    print("Context manager automatically cleaned up\n")


# ------------------------------------
# Example 3: Advanced Usage (Custom Event Loop)
# ------------------------------------
def advanced_usage_example():
    """
    Advanced usage where you manage your own event loop.
    Useful when integrating with existing async applications.
    """
    print("=== Advanced Usage Example ===")
    
    # Create and start your own event loop
    main_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=BatchScheduler.start_loop,  # Use the static method
        args=(main_loop,), 
        daemon=True
    )
    thread.start()
    
    # Create scheduler with your event loop
    scheduler = BatchScheduler(main_loop=main_loop)
    
    analysis_inputs = [
        "Machine learning adoption trends",
        "Cloud computing market analysis",
        "Blockchain technology impact"
    ]
    
    start_time = time.time()
    
    results = scheduler.forward(
        SimpleAnalyzer,
        num_workers=2,
        batch_size=2,
        dataset=analysis_inputs,
        temperature=0.3,  # Custom parameters
        max_tokens=200
    )
    
    print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    for i, result in enumerate(results):
        print(f"  Analysis {i+1}: {result}")
    
    # Clean up your custom event loop
    BatchScheduler.close_loop(main_loop, thread)
    print("Custom event loop cleaned up\n")


# ------------------------------------
# Example 4: Multiple Schedulers
# ------------------------------------
def multiple_schedulers_example():
    """
    Using multiple schedulers with a shared event loop.
    """
    print("=== Multiple Schedulers Example ===")
    
    # Create a shared event loop
    shared_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=BatchScheduler.start_loop,
        args=(shared_loop,),
        daemon=True
    )
    thread.start()
    
    # Create multiple schedulers sharing the same loop
    risk_scheduler = BatchScheduler(main_loop=shared_loop)
    data_scheduler = BatchScheduler(main_loop=shared_loop)
    
    # Run different tasks
    risk_results = risk_scheduler.forward(
        RiskManager,
        num_workers=1,
        batch_size=2,
        dataset=["Investment A", "Investment B"]
    )
    
    data_results = data_scheduler.forward(
        DataProcessor,
        num_workers=1,
        batch_size=2,
        dataset=["Sales data", "Customer data"]
    )
    
    print("Risk results:", risk_results)
    print("Data results:", data_results)
    
    # Clean up shared loop
    BatchScheduler.close_loop(shared_loop, thread)
    print("Shared event loop cleaned up\n")


# ------------------------------------
# Main execution
# ------------------------------------
if __name__ == "__main__":
    # Run all examples
    simple_usage_example()
    context_manager_example()
    advanced_usage_example()
    multiple_schedulers_example()