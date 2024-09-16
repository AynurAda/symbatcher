# symbatcher

## Description
Symbatcher is a tool that leverages the symbolicai library to provide efficient batch scheduling capabilities.

## Installation

To use symbatcher, you first need to install the symbolicai library. You can do this using pip:

```bash
pip install symbolicai
```

## Usage

```python
from symai import Expression
from src.func import BatchScheduler
from symai.functional import EngineRepository 
```python

Define a simple Expression

```python
class TestExpression(Expression):
    def forward(self, input, kwargs):
    return self.Symbol(input).query("Summarize this input", kwargs)

```python
Set up your engine (this example uses a mock engine)
```python

rom your_engine_module import MockGPTXChatEngine
engine = MockGPTXChatEngine()
EngineRepository.register("neurosymbolic", engine_instance=engine, allow_engine_override=True)

```python
Prepare your inputs
inputs = ["test1", "test2", "test3"]
```python

Create and run the BatchScheduler

```python
scheduler = BatchScheduler(TestExpression, num_workers=2, engine=engine, dataset=inputs)
results = scheduler.run()

for i, result in enumerate(results, 1):
    print(f"Result {i}: {result}")
```python