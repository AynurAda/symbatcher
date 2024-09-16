import time
import itertools
from typing import List, Any, Tuple 
import pytest

from symai import Expression, Symbol
from symai.backend.base import BatchEngine
from src.func import BatchScheduler
from symai.functional import EngineRepository   


class TestExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        res = Symbol(input).query("Summarize this input", **kwargs)
        return res.value

class NestedExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr = TestExpression()
    
    def forward(self, input, **kwargs):
        nested_result = self.nested_expr(input, **kwargs)
        return Symbol(nested_result).query("Elaborate on this result", **kwargs)

class DoubleNestedExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr1 = TestExpression()
        self.nested_expr2 = NestedExpression()
    
    def forward(self, input, **kwargs):
        result1 = self.nested_expr1(input, **kwargs)
        result2 = self.nested_expr2(input, **kwargs)
        return Symbol(f"{result1} and {result2}").query("Combine these results", **kwargs)

class ConditionalExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr = NestedExpression()
    
    def forward(self, input, **kwargs):
        if len(input) > 10:
            return Symbol(input).query("Analyze this long input", **kwargs)
        else:
            return Symbol(input).query("Briefly comment on this short input", **kwargs)

class SlowExpression(Expression):
    def __init__(self, delay=5, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
    
    def forward(self, input, **kwargs):
        time.sleep(self.delay)
        return Symbol(input).query(f"Process this input after a {self.delay} second delay", **kwargs)

class DoubleNestedExpressionSlow(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr1 = TestExpression()
        self.nested_expr2 = NestedExpression()
        self.slow_expr = SlowExpression()
    
    def forward(self, input, **kwargs):
        result1 = self.nested_expr1(input, **kwargs)
        result2 = self.nested_expr2(input, **kwargs)
        slow_result = self.slow_expr(input, **kwargs)
        return Symbol(f"{result1}, {result2}, and {slow_result}").query("Synthesize these results", **kwargs)

class ConditionalSlowExpression(Expression):
    def __init__(self, delay=5, threshold=10, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.threshold = threshold
        self.slow_expr = SlowExpression(delay=delay)
    
    def forward(self, input, **kwargs):
        if len(input) > self.threshold:
            return Symbol(self.slow_expr(input, **kwargs)).query("Analyze this slow-processed long input", **kwargs)
        else:
            return Symbol(input).query("Quickly process this short input", **kwargs)

class MockGPTXChatEngine(BatchEngine):
    def __init__(self):
        super().__init__()
        self.response_template = "This is a mock response for input: {}"
        self.model = "mock_model"
        self.max_tokens = 1000
        self.allows_batching = True

    def forward(self, arguments):
        responses = []
        metadata_list = []
        for argument in arguments:
            input_data = argument.prop.processed_input
            mock_response = self.response_template.format(input_data)
            mock_response = f"{mock_response}"
            responses.append(mock_response)
            
            individual_metadata = {
                "usage": {
                    "total_tokens": len(mock_response.split()),   
                    "prompt_tokens": len(input_data.split()),
                    "completion_tokens": len(mock_response.split()) - len(input_data.split())
                }
            }
            metadata_list.append(individual_metadata)
        
        return responses, metadata_list

    def prepare(self, argument):
        pass

engine = MockGPTXChatEngine()
EngineRepository.register("neurosymbolic", engine_instance=engine, allow_engine_override=True)


@pytest.fixture
def mock_engine():
    return engine

def test_simple_batch(mock_engine):
    expr = TestExpression
    inputs = ["test1", "test2", "test3"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"test{i}" in str(result)
        assert "Summarize this input" in str(result)

@pytest.mark.timeout(1)   
def test_nested_batch(mock_engine):
    expr = NestedExpression
    inputs = ["nested1", "nested2"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Elaborate on this result" in str(result)
        assert "Summarize this input" in str(result)

@pytest.mark.timeout(1)   
def test_conditional_batch(mock_engine):
    expr = ConditionalExpression
    inputs = ["short", "this is a long input"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    assert "Briefly comment on this short input" in str(results[0])
    assert "Analyze this long input" in str(results[1])

def test_slow_batch(mock_engine):
    expr = SlowExpression  
    inputs = ["slow1", "slow2"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    for i, result in enumerate(results, 1):
        assert f"slow{i}" in str(result)
        assert "Process this input after a 5 second delay" in str(result)

@pytest.mark.timeout(1)  
def test_double_nested_slow_batch(mock_engine):
    expr = DoubleNestedExpressionSlow
    inputs = ["input1", "input2"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    for i, result in enumerate(results, 1):
        assert f"input{i}" in str(result)
        assert "Synthesize these results" in str(result)
 
def test_simple_batch_variations(mock_engine):
    expr = TestExpression
    inputs = ["test1", "test2", "test3", "test4", "test5", "test6"]
    
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 6
    for i, result in enumerate(results, 1):
        assert f"test{i}" in str(result)
        assert "Summarize this input" in str(result)
    
    #Test with batch_size=3 and num_workers=2
    scheduler = BatchScheduler(expr, num_workers=2, batch_size=3, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 6
    for i, result in enumerate(results, 1):
        assert f"test{i}" in str(result)
        assert "Summarize this input" in str(result)

def test_nested_batch_variations(mock_engine):
    expr = NestedExpression
    inputs = ["nested1", "nested2", "nested3", "nested4"]
    
    # Test with batch_size=1 and num_workers=4
    scheduler = BatchScheduler(expr, num_workers=4, batch_size=1, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 4
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Elaborate on this result" in str(result)
    
    # Test with batch_size=4 and num_workers=1
    scheduler = BatchScheduler(expr, num_workers=1, batch_size=4, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 4
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Elaborate on this result" in str(result)

def test_conditional_batch_variations(mock_engine):
    expr = ConditionalExpression
    inputs = ["short", "this is a long input", "short+", "yet another long input"]

    # Test with batch_size=2 and num_workers=2
    scheduler = BatchScheduler(expr, num_workers=2, batch_size=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 4
    assert "Briefly comment on this short input" in str(results[0]), f"Unexpected result for 'short': {results[0]}"
    assert "Analyze this long input" in str(results[1]), f"Unexpected result for 'this is a long input': {results[1]}"
    assert "Briefly comment on this short input" in str(results[2]), f"Unexpected result for 'another short': {results[2]}"
    assert "Analyze this long input" in str(results[3]), f"Unexpected result for 'yet another long input': {results[3]}"

def test_slow_batch_variations(mock_engine):
    expr = SlowExpression
    inputs = ["slow1", "slow2", "slow3", "slow4", "slow5"]
    
    # Test with batch_size=2 and num_workers=3
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 5
    for i, result in enumerate(results, 1):
        assert f"slow{i}" in str(result)
        assert "Process this input after a 5 second delay" in str(result)
    
    # Test with batch_size=5 and num_workers=1
    scheduler = BatchScheduler(expr, num_workers=1, batch_size=5, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 5
    for i, result in enumerate(results, 1):
        assert f"slow{i}" in str(result)
        assert "Process this input after a 5 second delay" in str(result)

def test_double_nested_slow_batch_variations(mock_engine):
    expr = DoubleNestedExpressionSlow
    inputs = ["input1", "input2", "input3"]
    
    # Test with batch_size=1 and num_workers=3
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=1, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"input{i}" in str(result)
        assert "Synthesize these results" in str(result)
    
    # Test with batch_size=3 and num_workers=1
    scheduler = BatchScheduler(expr, num_workers=1, batch_size=3, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"input{i}" in str(result)
        assert "Synthesize these results" in str(result)

class RandomErrorExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error_pattern =[False, True, False, True, False]
        self.counter = itertools.cycle(self.error_pattern)
    
    def forward(self, input, **kwargs):
        if next(self.counter):
            raise ValueError("Simulated expression error")
        return Symbol(input).query("Process this input without error", **kwargs)

class MockRandomErrorEngine(MockGPTXChatEngine):
    def __init__(self):
        super().__init__()
        self.response_template = "This is a mock response for input: {} from error engine"

    def forward(self, arguments):
        responses = []
        metadata_list = []
        
        # Check if any input in the batch should trigger an error
        if any("error" in argument.prop.processed_input for argument in arguments):
            raise ValueError("Simulated engine error for the entire batch")
        
        for argument in arguments:
            input_data = argument.prop.processed_input
            
            mock_response = self.response_template.format(input_data)
            responses.append(mock_response)
            
            individual_metadata = {
                "usage": {
                    "total_tokens": 100,
                    "prompt_tokens": len(input_data.split()),
                    "completion_tokens": 100 - len(input_data.split())
                }
            }
            metadata_list.append(individual_metadata)
        
        return responses, metadata_list

@pytest.fixture
def mock_random_error_engine():
    return MockRandomErrorEngine()

def test_expression_error_handling(mock_engine):
    expr = RandomErrorExpression
    inputs = ["test1", "error", "test3", "error", "test5"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 5
    assert "Process this input without error" in str(results[0])
    assert isinstance(results[1], ValueError)
    assert "Process this input without error" in str(results[2])
    assert isinstance(results[3], ValueError)
    assert "Process this input without error" in str(results[4])

def test_engine_error_handling(mock_random_error_engine):
    expr = TestExpression
    inputs = ["test1", "error", "test3", "test4", "error"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_random_error_engine, dataset=inputs, batch_size=2)
    results = scheduler.run()
    assert len(results) == 5
    assert results[0]=="Simulated engine error for the entire batch"
    assert results[1]=="Simulated engine error for the entire batch"
    assert "Summarize this input" in str(results[2])
    assert "Summarize this input" in str(results[3])
    assert results[4]=="Simulated engine error for the entire batch"


def test_double_nested_batch(mock_engine):
    expr = DoubleNestedExpression
    inputs = ["nested1", "nested2", "nested3"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Combine these results" in str(result)
        assert "Summarize this input" in str(result)
        assert "Elaborate on this result" in str(result)