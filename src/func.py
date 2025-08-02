import concurrent.futures
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from symai.core_ext import bind
from symai import Expression
from symai.functional import EngineRepository
import time
import asyncio

lgr = logging.getLogger()
lgr.setLevel(logging.CRITICAL)

class BatchScheduler(Expression):
    """
    A class for scheduling and executing batch operations with Expressions from symbolicai.

    This scheduler manages the concurrent execution of symbolicai Expressions in batches,
    utilizing multiple workers and an external engine for processing.
    """

    def __init__(self, main_loop=None): 
        """Initialize the BatchScheduler with optional event loop.
        
        Args:
            main_loop: Optional asyncio event loop. If not provided, one will be created automatically.
        """
        super().__init__()
        repository = EngineRepository()
        repository.get('neurosymbolic').__setattr__("executor_callback",self.executor_callback)
        self.id_queue = []
        self.current_batch_responses_received = 0
        
        # Track if we created the loop internally
        self._owns_loop = main_loop is None
        self.loop_thread = None
        
        if main_loop is None:
            # Create our own event loop
            self.main_loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(
                target=self.start_loop, 
                args=(self.main_loop,),
                daemon=True,
                name="BatchScheduler-EventLoop"
            )
            self.loop_thread.start()
            # Give the loop time to start
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
        """
        Stop an event-loop that's running in another thread and close it cleanly.
        """
        if not loop.is_closed():
            # Step 1 – ask the loop to stop
            loop.call_soon_threadsafe(loop.stop)

            # Step 2 – wait (briefly) for the thread to exit the run_forever loop
            thread.join(timeout=2)

            # Step 3 – close the loop (must happen in the thread that created it)
            #          so we schedule the close on that thread:
            def _close() -> None:
                loop.close()

            loop.call_soon_threadsafe(_close)
            lgr.info("Loop stop requested and close scheduled.")
        else:
            lgr.info("Loop already closed.")
    
    def cleanup(self):
        """Clean up resources, including the event loop if we created it."""
        if self._owns_loop and self.loop_thread and self.main_loop:
            self.close_loop(self.main_loop, self.loop_thread)
            self.loop_thread = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.cleanup()
        return False

    def engine_execution(self, batch):
        @bind(engine="neurosymbolic", property="__call__")
        def engine(self, *args, **kwargs):
            """
            Engine method that will be bound to the neurosymbolic engine's __call__.
            Must accept arguments to match the engine's interface.
            """
            pass
        
        # The bind decorator returns the bound method directly
        # Just call it synchronously - engines in symbolicai are not async
        llm_batch_responses = engine()(batch)
        llm_batch_responses = [(resp[0] if isinstance(resp[0], list) else [resp[0]], resp[1]) for resp in llm_batch_responses]
        return llm_batch_responses

    def single_expression(self, data_point: Any, **kwargs) -> Any:
        """
        Execute the symbolicai Expression for a single data point.

        Args:
            data_point (Any): The data point to process through the Expression.

        Returns:
            Any: The result of the Expression execution.
        """
        expr = self.expr
        try:
            res = expr(input = data_point, **kwargs)
            self.pending_tasks -= 1
            return res
        except Exception as e:
            print(f"Data point {data_point} generated an exception: {str(e)}")
            self.pending_tasks -= 1
            return e
        finally:
            if self.pending_tasks==0:
                self.all_batches_complete.set()
    
    def release_batch_for_execution_if_full(self):
        enough_calls_in_line = (len(self.llm_calls_queue) >= self.batch_size)
        last_incomplete_batch = (len(self.llm_calls_queue) > 0 and self.pending_tasks < self.batch_size)  
        if enough_calls_in_line or last_incomplete_batch or self.pending_tasks == 0:
            print(f"DEBUG - RELEASING BATCH - Reason: {'enough_calls' if enough_calls_in_line else 'incomplete_batch' if last_incomplete_batch else 'no_pending_tasks'}")
            self.batch_ready_for_exec.set()

    def schedule_engine_call(self, engine_call):
        self.llm_calls_queue.append(engine_call)
        engine_call_id = id(engine_call)
        self.id_queue.append(engine_call_id)
        self.llm_responses[engine_call_id] = None
        self.llm_response_ready[engine_call_id] = threading.Event()
        return engine_call_id
    
    def get_resp_from_engine(self, engine_call_id):
        self.llm_response_ready[engine_call_id].wait()
        with self.lock:
            llm_response = self.llm_responses.pop(engine_call_id)
            del self.llm_response_ready[engine_call_id]
        return llm_response


    def executor_callback(self, engine_call: Any) -> Any:
        """
        Callback function for the executor to handle arguments and responses from the Expression.

        This method is called by the symbolicai Expression during execution.

        Args:
            engine_call (Any): The engine_call generated by the Expression to be processed.

        Returns:
            Any: The processed response for the given engine_call.
        """
        with self.lock:
            engine_call_id = self.schedule_engine_call(engine_call)
        llm_response = self.get_resp_from_engine(engine_call_id)
        self.current_batch_responses_received += 1 
        return llm_response   
 
    def get_next_batch(self):
        """Get the next batch of queries from the queue."""
        with self.lock:
            current_batch = self.llm_calls_queue[:self.batch_size]
            self.llm_calls_queue = self.llm_calls_queue[self.batch_size:]
            current_batch_ids = self.id_queue[:self.batch_size]
            self.id_queue = self.id_queue[self.batch_size:]
        return current_batch, current_batch_ids
    
    def distribute_engine_responses_to_threads(self, llm_batch_responses, current_batch_ids):
        """Process and store responses for each item in the batch."""
        with self.lock:
            for llm_response, arg_id in zip(llm_batch_responses, current_batch_ids):   
                self.llm_responses[arg_id] = llm_response
                self.llm_response_ready[arg_id].set()

    def execute_queries(self) -> None:
      print("Starting execute_queries loop")
      while not self.all_batches_complete.is_set():
          # Keep checking until the queue is ready for execution
          while not self.batch_ready_for_exec.is_set() and not self.all_batches_complete.is_set():
              self.release_batch_for_execution_if_full()
              time.sleep(0.01)  # Avoid tight CPU loop

          if self.pending_tasks > 0 and self.batch_ready_for_exec.is_set():
              self.batch_ready_for_exec.clear()

              self.current_batch_responses_received = 0
              current_batch, current_batch_ids = self.get_next_batch()
              if not current_batch:
                  continue

              llm_batch_responses = self.engine_execution(current_batch)
              self.distribute_engine_responses_to_threads(llm_batch_responses, current_batch_ids)

              while self.current_batch_responses_received < len(current_batch):
                  time.sleep(0.01)

          if self.pending_tasks == 0:
              self.all_batches_complete.set()
              break
      print(f"Exiting execute_queries - Queue: {len(self.llm_calls_queue)}, Pending: {self.pending_tasks}")

    def forward(self, expr: Expression, num_workers: int, dataset: List[Any], batch_size: int = 5, **kwargs) -> List[
        Any]:
        """
        Run the batch scheduling process for symbolicai Expressions.

        Args:
            expr (Expression): The symbolicai Expression to be executed.
            num_workers (int): The number of worker threads to use.
            dataset (List[Any]): The list of data points to process through the Expression.
            batch_size (int, optional): The size of each batch. Defaults to 5.

        Returns:
            List[Any]: A list of final results for each data point in the dataset.
        """
        # Create a new dictionary with id as key and datapoint as value
        dataset_dict = {i: datapoint for i, datapoint in enumerate(dataset)}

        # Rest of the implementation
        self.num_workers = num_workers
        self.dataset = list(dataset_dict.values())
        self.results = {}
        self.llm_calls_queue = []
        self.lock = threading.Lock()
        self.batch_size = min(batch_size, len(self.dataset) if self.dataset else 1, num_workers)
        self.batch_ready_for_exec = threading.Event()
        self.all_batches_complete = threading.Event()
        self.llm_responses = {}
        self.llm_response_ready = {}
        self.pending_tasks = len(self.dataset)
        self.expr = expr(**kwargs)

        query_thread = threading.Thread(target=self.execute_queries)
        query_thread.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_data = {executor.submit(self.single_expression, datapoint, **kwargs): (i, datapoint)
                              for i, datapoint in dataset_dict.items()}

            concurrent.futures.wait(future_to_data.keys())

            for future in future_to_data:
                data_id, datapoint = future_to_data[future]
                try:
                    final_result = future.result()
                    self.results[data_id] = final_result
                except Exception as exc:
                    logging.error(f'batch {data_id} generated an exception: {exc}')

        query_thread.join()

        return [self.results.get(i) for i in range(len(dataset))]
