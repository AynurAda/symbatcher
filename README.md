# symbatcher

## Description
Symbatcher is a tool that leverages the symbolicai library to provide efficient batch scheduling capabilities.
Currently only 1 engine at a time is supported. Engine switching to be introduced in a future release.

## Installation

To use symbatcher, you first need to install the symbolicai library. You can do this using pip:

```bash
pip install symbolicai
```

## Usage

```python
from symai import Import
from symai import Expression

##Define a simple Expression
class TestExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, input, kwargs):
        return self.Symbol(input).query("Summarize this input", kwargs)


module = Import('AynurAda/symbatcher', expr = TestExpression, dataset = ["test1", "test2", "test3"], num_workers=1, batch_size=1)
 



 
