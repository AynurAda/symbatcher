"""
Simple test for AsyncGeminiXBatchEngine with rate limiting
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symai import Symbol, Expression
from symai.functional import EngineRepository
from src.func import BatchScheduler
from src.engines.async_gemini_reasoning_rate_limited import AsyncGeminiXBatchEngine


class SimpleTask(Expression):
    def __init__(self):
        super().__init__()

    def forward(self, input: str, **kwargs) -> Symbol:
        return Symbol().query(input)


if __name__ == '__main__':
    # Create and register async Gemini engine
    engine = AsyncGeminiXBatchEngine(
        model='gemini-1.5-flash',
        rate_limits='default'  # or None for no rate limiting, or custom dict
    )
    EngineRepository.register('neurosymbolic', engine)

    # Create batch scheduler
    batch_scheduler = BatchScheduler()

    # Test prompts
    prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Name a color.",
        "What is 10*5?",
        "Name an animal.",
    ]


    # Run batch
    results = batch_scheduler(
        SimpleTask,
        batch_size=5,
        num_workers=5,
        dataset = prompts
    )

    # Print results
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")
