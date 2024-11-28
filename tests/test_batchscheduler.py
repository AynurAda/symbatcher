import os
import sys
import time

import pytest
from symai import Expression, Symbol
from symai.backend.base import BatchEngine
from symai.functional import EngineRepository

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.func import BatchScheduler

# delay for tests with slow expressions
delay = 1

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

mock_engine = MockGPTXChatEngine()
EngineRepository.register("neurosymbolic", engine_instance=mock_engine, allow_engine_override=True)

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
    def __init__(self, delay=delay, **kwargs):
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
    def __init__(self, delay=delay, threshold=10, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.threshold = threshold
        self.slow_expr = SlowExpression(delay=delay)
    
    def forward(self, input, **kwargs):
        if len(input) > self.threshold:
            return Symbol(self.slow_expr(input, **kwargs)).query("Analyze this slow-processed long input", **kwargs)
        else:
            return Symbol(input).query("Quickly process this short input", **kwargs)

# 1
@pytest.mark.timeout(5)
def test_simple_batch():
    expr = TestExpression
    inputs = ["test1", "test2", "test3"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"test{i}" in str(result)
        assert "Summarize this input" in str(result)
# 2
@pytest.mark.timeout(5)
def test_nested_batch():
    expr = NestedExpression
    inputs = ["nested1", "nested2"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    assert len(results) == 2
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Elaborate on this result" in str(result)
        assert "Summarize this input" in str(result)

# 3
@pytest.mark.timeout(5)
def test_conditional_batch():
    expr = ConditionalExpression
    inputs = ["short", "this is a long input"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    assert len(results) == 2
    assert "Briefly comment on this short input" in str(results[0])
    assert "Analyze this long input" in str(results[1])

# 4
@pytest.mark.timeout(5)
def test_slow_batch():
    expr = SlowExpression
    inputs = ["slow1", "slow2"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    assert len(results) == 2
    for i, result in enumerate(results, 1):
        assert f"slow{i}" in str(result)
        assert f"Process this input after a {delay} second delay" in str(result)

# 5
@pytest.mark.timeout(5)
def test_double_nested_slow_batch():
    expr = DoubleNestedExpressionSlow
    inputs = ["input1", "input2"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    assert len(results) == 2
    for i, result in enumerate(results, 1):
        assert f"input{i}" in str(result)
        assert "Synthesize these results" in str(result)

# 6
@pytest.mark.timeout(5)
def test_simple_batch_variations():
    expr = TestExpression
    inputs = ["test1", "test2", "test3", "test4", "test5", "test6"]
    
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=3, dataset=inputs, batch_size=2)
    assert len(results) == 6
    for i, result in enumerate(results, 1):
        assert f"test{i}" in str(result)
        assert "Summarize this input" in str(result)
    
    # Test with batch_size=3 and num_workers=2
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs, batch_size=3)
    assert len(results) == 6
    for i, result in enumerate(results, 1):
        assert f"test{i}" in str(result)
        assert "Summarize this input" in str(result)

# 7
@pytest.mark.timeout(5)
def test_nested_batch_variations():
    expr = NestedExpression
    inputs = ["nested1", "nested2", "nested3", "nested4"]
    
    # Test with batch_size=1 and num_workers=4
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=4, dataset=inputs, batch_size=1)
    assert len(results) == 4
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Elaborate on this result" in str(result)
    
    # Test with batch_size=4 and num_workers=1
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=1, dataset=inputs, batch_size=4)
    assert len(results) == 4
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Elaborate on this result" in str(result)

@pytest.mark.timeout(5)
def test_conditional_batch_variations():
    expr = ConditionalExpression
    inputs = ["short", "this is a long input", "short+", "yet another long input"]

    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs, batch_size=2)
    assert len(results) == 4
    assert "Briefly comment on this short input" in str(results[0])
    assert "Analyze this long input" in str(results[1])
    assert "Briefly comment on this short input" in str(results[2])
    assert "Analyze this long input" in str(results[3])

@pytest.mark.timeout(5)
def test_slow_batch_variations():
    expr = SlowExpression
    inputs = ["slow1", "slow2", "slow3", "slow4", "slow5"]
    
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=3, dataset=inputs, batch_size=2)
    assert len(results) == 5
    for i, result in enumerate(results, 1):
        assert f"slow{i}" in str(result)
        assert f"Process this input after a {delay} second delay" in str(result)
    
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=1, dataset=inputs, batch_size=5)
    assert len(results) == 5
    for i, result in enumerate(results, 1):
        assert f"slow{i}" in str(result)
        assert f"Process this input after a {delay} second delay" in str(result)

@pytest.mark.timeout(5)
def test_double_nested_slow_batch_variations():
    expr = DoubleNestedExpressionSlow
    inputs = ["input1", "input2", "input3"]
    
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=3, dataset=inputs, batch_size=1)
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"input{i}" in str(result)
        assert "Synthesize these results" in str(result)
    
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=1, dataset=inputs, batch_size=3)
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"input{i}" in str(result)
        assert "Synthesize these results" in str(result)


@pytest.mark.timeout(5)
def test_double_nested_batch():
    expr = DoubleNestedExpression
    inputs = ["nested1", "nested2", "nested3"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert f"nested{i}" in str(result)
        assert "Combine these results" in str(result)
        assert "Summarize this input" in str(result)
        assert "Elaborate on this result" in str(result)


class FaultyExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input, **kwargs):
        # Introduce a bug before calling executor_callback
        raise ValueError("Intentional exception before calling executor_callback")
        # The code never reaches executor_callback due to the exception


@pytest.mark.timeout(5)
def test_faulty_expression():
    expr = FaultyExpression
    inputs = ["input1", "input2", "input3", "input4", "input5", "input6"]
    scheduler = BatchScheduler()

    # Run the scheduler and expect exceptions in the results
    results = scheduler(expr, num_workers=2, dataset=inputs)

    # Verify that results contain exceptions and the scheduler didn't get stuck
    assert len(results) == 6
    for result in results:
        assert isinstance(result, Exception)
        assert "Intentional exception before calling executor_callback" in str(result)
