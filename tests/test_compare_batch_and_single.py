import itertools
import os
import sys
import time

import pytest
from symai import Expression, Symbol
from symai.backend.base import BatchEngine
from symai.functional import EngineRepository

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))
from src.func import BatchScheduler

class TestExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        res = Symbol(input).query("Follow this instruction: ", **kwargs)
        return res.value

def test_expression_with_and_without_scheduler():
    expr = TestExpression()
    input_text = "Write an essay about AI"
    kwargs = {"temperature": 0}
    
    # Test without scheduler
    direct_result = expr(input_text, **kwargs)
    direct_result2 = expr(input_text, **kwargs)

    # Test with scheduler
    scheduler = BatchScheduler()
    scheduled_results = scheduler(TestExpression, num_workers=1, dataset=[input_text], **kwargs)
    scheduled_result = scheduled_results[0]
    
    # Both should contain the input prompt and have temperature=0 applied
    assert "Write an essay about AI" in str(direct_result)
    assert "Write an essay about AI" in str(scheduled_result)
    assert "Summarize this input" in str(direct_result)
    assert "Summarize this input" in str(scheduled_result)