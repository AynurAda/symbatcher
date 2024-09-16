# symbatcher

## Description
Symbatcher is a tool that leverages the symbolicai library to provide efficient batch scheduling capabilities.

## Installation

To use symbatcher, you first need to install the symbolicai library. You can do this using pip:

```bash
pip install symbolicai
```

## Usage
python
from symai import Expression
from src.func import BatchScheduler
from symai.functional import EngineRepository
Define a simple Expression
class TestExpression(Expression):
def forward(self, input, kwargs):
return self.Symbol(input).query("Summarize this input", kwargs)
Set up your engine (this example uses a mock engine)
from your_engine_module import MockGPTXChatEngine
engine = MockGPTXChatEngine()
EngineRepository.register("neurosymbolic", engine_instance=engine, allow_engine_override=True)
Prepare your inputs
inputs = ["test1", "test2", "test3"]
Create and run the BatchScheduler
scheduler = BatchScheduler(TestExpression, num_workers=2, engine=engine, dataset=inputs)
results = scheduler.run()
Process the results
for i, result in enumerate(results, 1):
print(f"Result {i}: {result}")
