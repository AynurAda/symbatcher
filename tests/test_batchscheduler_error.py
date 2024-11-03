import itertools
import os
import sys
import time

import pytest
from symai import Expression, Symbol
from symai.backend.base import BatchEngine
from symai.functional import EngineRepository

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.func import BatchScheduler


class MockRandomErrorEngine(BatchEngine):
    def __init__(self):
        super().__init__()
        self.response_template = "This is a mock response for input: {} from engine"

    def forward(self, arguments):
        responses = []
        metadata_list = []
        
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
    
    def prepare(self, argument):
        pass

mock_error_engine = MockRandomErrorEngine()
EngineRepository.register("neurosymbolic", engine_instance=mock_error_engine, allow_engine_override=True)

class TestExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        res = Symbol(input).query("Summarize this input", **kwargs)
        return res.value

class RandomErrorExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error_pattern =[False, True, False, True, False]
        self.counter = itertools.cycle(self.error_pattern)
    
    def forward(self, input, **kwargs):
        if next(self.counter):
            raise ValueError("Simulated expression error")
        return Symbol(input).query("Process this input without mistake", **kwargs)

 
def test_expression_error_handling():
    expr = RandomErrorExpression
    inputs = ["test1", "error", "test3", "error", "test5"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs)
    print(results)
    assert len(results) == 5
    assert "Process this input without mistake" in str(results[0])
    assert isinstance(results[1], ValueError)
    assert "Process this input without mistake" in str(results[2])
    assert isinstance(results[3], ValueError)
    assert "Process this input without mistake" in str(results[4])

def test_engine_error_handling():
    expr = TestExpression
    inputs = ["test1", "error", "test3", "test4", "error"]
    scheduler = BatchScheduler()
    results = scheduler(expr, num_workers=2, dataset=inputs, batch_size=2)
    assert len(results) == 5
    assert results[0]=="Simulated engine error for the entire batch"
    assert results[1]=="Simulated engine error for the entire batch"
    assert "Summarize this input" in str(results[2])
    assert "Summarize this input" in str(results[3])
    assert results[4]=="Simulated engine error for the entire batch"
