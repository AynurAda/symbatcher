import concurrent.futures
import logging
import threading
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from symai import Expression
from symai.functional import EngineRepository
from symai.core_ext import bind

lgr = logging.getLogger()
lgr.setLevel(logging.CRITICAL)

class MultiEngineBatchScheduler(Expression):
    """
    A multi-engine batch scheduler that extends BatchScheduler to support 
    batching across multiple specified engines.
    """
    
    def __init__(self, engines_to_batch: List[str] = None, main_loop=None):
        """
        Initialize the multi-engine batch scheduler.
        
        Args:
            engines_to_batch: List of engine names to enable batching for.
                            Default: ['neurosymbolic']
            main_loop: Optional asyncio event loop
        """
        super().__init__()
        
        # Default to neurosymbolic if not specified
        if engines_to_batch is None:
            engines_to_batch = ['neurosymbolic']
        
        self.engines_to_batch = engines_to_batch
        
        # Initialize repository and set callbacks
        repository = EngineRepository()
        for engine_name in engines_to_batch:
            try:
                engine = repository.get(engine_name)
                # Set our executor callback
                engine.__setattr__("executor_callback",
                                 lambda call, en=engine_name: self.executor_callback(call, en))
                lgr.info(f"Enabled batching for engine: {engine_name}")
            except Exception as e:
                lgr.warning(f"Failed to enable batching for {engine_name}: {e}")
        
        # Initialize same variables as single-engine version
        self.id_queue = []
        self.current_batch_responses_received = 0
        
        # Multi-engine specific: separate queues per engine
        self.engine_locks = {name: threading.Lock() for name in engines_to_batch}
        self.engine_queues = defaultdict(list)
        self.engine_responses = defaultdict(dict)
        self.engine_response_ready = defaultdict(dict)
        self.batch_ready_events = {}
        
        # Event loop management
        self._owns_loop = main_loop is None
        self.loop_thread = None
        
        if main_loop is None:
            self.main_loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(
                target=self.start_loop, 
                args=(self.main_loop,),
                daemon=True,
                name="MultiEngineBatchScheduler-EventLoop"
            )
            self.loop_thread.start()
            time.sleep(0.1)
        else:
            self.main_loop = main_loop
    
    @staticmethod
    def start_loop(loop: asyncio.AbstractEventLoop) -> None:
        """Run the given event-loop forever in the current thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    @staticmethod
    def close_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
        """Stop an event-loop that's running in another thread and close it cleanly."""
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            def _close() -> None:
                loop.close()
            loop.call_soon_threadsafe(_close)
            lgr.info("Loop stop requested and close scheduled.")
        else:
            lgr.info("Loop already closed.")
    
    def cleanup(self):
        """Clean up resources."""
        if self._owns_loop and self.loop_thread and self.main_loop:
            self.close_loop(self.main_loop, self.loop_thread)
            self.loop_thread = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def schedule_engine_call(self, engine_call, engine_name: str):
        self.engine_queues[engine_name].append(engine_call)
        engine_call_id = id(engine_call)
        # For multi-engine, we don't need a global id_queue
        self.engine_responses[engine_name][engine_call_id] = None
        self.engine_response_ready[engine_name][engine_call_id] = threading.Event()
        return engine_call_id
    
    def get_resp_from_engine(self, engine_call_id, engine_name: str):
        self.engine_response_ready[engine_name][engine_call_id].wait()
        with self.engine_locks[engine_name]:
            llm_response = self.engine_responses[engine_name].pop(engine_call_id)
            del self.engine_response_ready[engine_name][engine_call_id]
        return llm_response
    
    def executor_callback(self, engine_call: Any, engine_name: str) -> Any:
        """
        Callback function for the executor to handle arguments and responses from the Expression.
        
        This method is called by the symbolicai Expression during execution.
        
        Args:
            engine_call (Any): The engine_call generated by the Expression to be processed.
            engine_name (str): Which engine should process this call
        
        Returns:
            Any: The processed response for the given engine_call.
        """
        with self.engine_locks[engine_name]:
            engine_call_id = self.schedule_engine_call(engine_call, engine_name)
        llm_response = self.get_resp_from_engine(engine_call_id, engine_name)
        self.current_batch_responses_received += 1 
        return llm_response
    
    def release_batch_for_execution_if_full(self, engine_name: str):
        """Check if batch is ready for a specific engine."""
        with self.engine_locks[engine_name]:
            queue_size = len(self.engine_queues[engine_name])
            enough_calls = queue_size >= self.batch_size
            last_incomplete = queue_size > 0 and self.pending_tasks < self.batch_size
            no_more_tasks = self.pending_tasks == 0
            
            if enough_calls or last_incomplete or no_more_tasks:
                if engine_name not in self.batch_ready_events:
                    self.batch_ready_events[engine_name] = threading.Event()
                self.batch_ready_events[engine_name].set()
    
    def engine_execution(self, batch, engine_name: str):
        @bind(engine=engine_name, property="__call__")
        def engine(self, *args, **kwargs):
            """
            Engine method that will be bound to the specified engine's __call__.
            Must accept arguments to match the engine's interface.
            """
            pass
        
        # The bind decorator returns the bound method directly
        # Just call it synchronously - engines in symbolicai are not async
        llm_batch_responses = engine()(batch)
        llm_batch_responses = [(resp[0] if isinstance(resp[0], list) else [resp[0]], resp[1]) for resp in llm_batch_responses]
        return llm_batch_responses
    
    def get_next_batch(self, engine_name: str):
        """Get the next batch of queries from the queue for a specific engine."""
        with self.engine_locks[engine_name]:
            current_batch = self.engine_queues[engine_name][:self.batch_size]
            self.engine_queues[engine_name] = self.engine_queues[engine_name][self.batch_size:]
            # Extract IDs from the batch
            current_batch_ids = [id(call) for call in current_batch]
        return current_batch, current_batch_ids
    
    def distribute_engine_responses_to_threads(self, llm_batch_responses, current_batch_ids, engine_name: str):
        """Process and store responses for each item in the batch."""
        with self.engine_locks[engine_name]:
            for llm_response, arg_id in zip(llm_batch_responses, current_batch_ids):   
                self.engine_responses[engine_name][arg_id] = llm_response
                self.engine_response_ready[engine_name][arg_id].set()
    
    def execute_queries_for_engine(self, engine_name: str) -> None:
        """Execute queries for a specific engine."""
        print(f"Starting execute_queries for engine: {engine_name}")
        
        while not self.all_batches_complete.is_set():
            # Keep checking until the queue is ready for execution
            while not self.batch_ready_events.get(engine_name, threading.Event()).is_set() and not self.all_batches_complete.is_set():
                self.release_batch_for_execution_if_full(engine_name)
                time.sleep(0.01)  # Avoid tight CPU loop
            
            if self.pending_tasks > 0 and engine_name in self.batch_ready_events and self.batch_ready_events[engine_name].is_set():
                self.batch_ready_events[engine_name].clear()
                
                self.current_batch_responses_received = 0
                current_batch, current_batch_ids = self.get_next_batch(engine_name)
                if not current_batch:
                    continue
                
                llm_batch_responses = self.engine_execution(current_batch, engine_name)
                self.distribute_engine_responses_to_threads(llm_batch_responses, current_batch_ids, engine_name)
                
                while self.current_batch_responses_received < len(current_batch):
                    time.sleep(0.01)
            
            if self.pending_tasks == 0:
                self.all_batches_complete.set()
                break
        
        print(f"Exiting execute_queries for engine: {engine_name}")
    
    def single_expression(self, data_point: Any, **kwargs) -> Any:
        """Execute a single expression."""
        expr = self.expr
        try:
            res = expr(input=data_point, **kwargs)
            self.pending_tasks -= 1
            return res
        except Exception as e:
            print(f"Data point {data_point} generated an exception: {str(e)}")
            self.pending_tasks -= 1
            return e
        finally:
            if self.pending_tasks == 0:
                self.all_batches_complete.set()
    
    def forward(self, expr: Expression, num_workers: int, dataset: List[Any], 
                batch_size: int = 5, **kwargs) -> List[Any]:
        """
        Run the batch scheduling process for expressions.
        
        Args:
            expr: The Expression class to be executed
            num_workers: The number of worker threads to use
            dataset: The list of data points to process
            batch_size: The size of each batch (default: 5)
            **kwargs: Additional arguments passed to expressions
        
        Returns:
            List of results in the same order as dataset
        """
        # Create dataset dict with indices
        dataset_dict = {i: datapoint for i, datapoint in enumerate(dataset)}
        
        # Initialize shared state (matching original as much as possible)
        self.num_workers = num_workers
        self.dataset = list(dataset_dict.values())
        self.results = {}
        self.batch_size = min(batch_size, len(self.dataset) if self.dataset else 1, num_workers)
        self.all_batches_complete = threading.Event()
        self.pending_tasks = len(self.dataset)
        self.expr = expr(**kwargs)
        
        # Initialize per-engine queues and responses
        for engine_name in self.engines_to_batch:
            self.engine_queues[engine_name] = []
            self.engine_responses[engine_name] = {}
            self.engine_response_ready[engine_name] = {}
        
        # Start query threads for engines we're batching
        query_threads = []
        for engine_name in self.engines_to_batch:
            thread = threading.Thread(
                target=self.execute_queries_for_engine,
                args=(engine_name,),
                name=f"QueryThread-{engine_name}"
            )
            thread.start()
            query_threads.append(thread)
        
        # Process dataset with workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_data = {
                executor.submit(self.single_expression, datapoint, **kwargs): (i, datapoint)
                for i, datapoint in dataset_dict.items()
            }
            
            # Wait for all futures to complete
            concurrent.futures.wait(future_to_data.keys())
            
            # Collect results
            for future in future_to_data:
                data_id, datapoint = future_to_data[future]
                try:
                    final_result = future.result()
                    self.results[data_id] = final_result
                except Exception as exc:
                    logging.error(f'batch {data_id} generated an exception: {exc}')
                    self.results[data_id] = None
        
        # Wait for query threads to finish
        for thread in query_threads:
            thread.join()
        
        # Return results in order
        return [self.results.get(i) for i in range(len(dataset))]