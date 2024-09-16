import concurrent.futures
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from symai import Expression

lgr = logging.getLogger()
lgr.setLevel(logging.CRITICAL)

class BatchScheduler:
    """
    A class for scheduling and executing batch operations with Expressions from symbolicai.

    This scheduler manages the concurrent execution of symbolicai Expressions in batches,
    utilizing multiple workers and an external engine for processing.
    """

    def __init__(self, expr: Expression, num_workers: int, engine: Callable, dataset: List[Any], batch_size: int = 5):
        """
        Initialize the BatchScheduler for symbolicai Expressions.

        Args:
            expr (Expression): The symbolicai Expression to be executed.
            num_workers (int): The number of worker threads to use.
            engine (Callable): The engine function for processing batches of Expression results.
            dataset (List[Any]): The list of data points to process through the Expression.
            batch_size (int, optional): The size of each batch. Defaults to 5.
        """
        self.num_workers: int = num_workers
        self.engine: Callable = engine
        self.dataset: List[Any] = dataset
        self.results: Dict[Any, Any] = {}
        self.arguments: List[Any] = []
        self.lock: threading.Lock = threading.Lock()
        self.batch_size: int = min(batch_size, len(dataset) if dataset else 1, num_workers)
        self.batch_ready: threading.Event = threading.Event()
        self.processing_complete: threading.Event = threading.Event()
        self.llm_responses: Dict[int, Optional[Any]] = {}
        self.llm_response_ready: Dict[int, threading.Event] = {}
        self.pending_tasks: int = len(self.dataset)
        self.expr: Expression = expr
        self.pending_tasks_update: threading.Event = threading.Event()
 
    def single_expression(self, data_point: Any) -> Any:
        """
        Execute the symbolicai Expression for a single data point.

        Args:
            data_point (Any): The data point to process through the Expression.

        Returns:
            Any: The result of the Expression execution.
        """
        expr = self.expr
        try:
            return expr(data_point, executor_callback=self.executor_callback)
        except Exception as e:
            print(f"Data point {data_point} generated an exception: {str(e)}")
            return e   
 
    def executor_callback(self, argument: Any) -> Any:
        """
        Callback function for the executor to handle arguments and responses from the Expression.

        This method is called by the symbolicai Expression during execution.

        Args:
            argument (Any): The argument generated by the Expression to be processed.

        Returns:
            Any: The processed response for the given argument.
        """
        with self.lock:
            self.arguments.append(argument)
            arg_id = id(argument)
            if arg_id not in self.llm_responses.keys():
                self.llm_responses[arg_id] = None
                self.llm_response_ready[arg_id] = threading.Event()
            if len(self.arguments) >= self.batch_size or self.pending_tasks < self.batch_size:
                self.batch_ready.set()
        self.llm_response_ready[arg_id].wait()
        with self.lock:
            llm_response = self.llm_responses.pop(arg_id)
            del self.llm_response_ready[arg_id]
        return llm_response   
 
    def execute_queries(self) -> None:
        """
        Execute queries in batches using the engine.

        This method runs in a separate thread and processes batches of arguments
        generated by the symbolicai Expressions.
        """
        while not self.processing_complete.is_set() or self.arguments:
            self.batch_ready.wait()
            self.batch_ready.clear()
            with self.lock:
                current_arguments = self.arguments[:self.batch_size]
                self.arguments = self.arguments[self.batch_size:]      
            if current_arguments:
                llm_batch_responses = self.engine(current_arguments)
                llm_batch_responses = [(resp[0] if isinstance(resp[0], list) else [resp[0]], resp[1]) for resp in llm_batch_responses]
                for arg, llm_response in zip(current_arguments, llm_batch_responses):
                    with self.lock:
                        arg_id = id(arg)
                        self.llm_responses[arg_id] = llm_response
                        self.llm_response_ready[arg_id].set()
            if self.arguments:
                self.batch_ready.set()
 
    def run(self) -> List[Any]:
        """
        Run the batch scheduling process for symbolicai Expressions.

        This method starts the query execution thread and manages the concurrent
        processing of data points through the Expression.

        Returns:
            List[Any]: A list of final results for each data point in the dataset.
        """
        query_thread = threading.Thread(target=self.execute_queries)
        query_thread.start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_data = {executor.submit(self.single_expression, data_point): data_point for data_point in self.dataset}
            for future in concurrent.futures.as_completed(future_to_data):
                data_point = future_to_data[future]
                try:
                    final_result = future.result()
                    self.results[data_point] = final_result
                except Exception as exc:
                    print(f'Data point {data_point} generated an exception: {exc}')
                finally:
                    self.pending_tasks -= 1
                    self.batch_ready.set()
        self.processing_complete.set()
        print("processing complete")
        self.batch_ready.set()   
        query_thread.join()
        return [self.results.get(data_point) for data_point in self.dataset]