# Multi-Engine Architecture Analysis for BatchScheduler

## Engine Routing Flow in SymbolicAI

Based on my analysis, here's how requests are routed to the right engine:

### 1. **Decorator-Based Engine Selection**
SymbolicAI uses specific decorators in `core.py` that explicitly specify which engine to use:
- `@few_shot`, `@zero_shot` → `engine='neurosymbolic'` (line 118)
- `@expression` → `engine='symbolic'` (line 799) 
- `@search` → `engine='search'` (line 1131)
- `@file` → `engine='files'` (line 1157)
- `@embed` → `engine='embedding'` (line 1181)
- `@draw` → `engine='drawing'` (line 1221)
- `@clip` → `engine='text_vision'` (line 1245)
- `@ocr` → `engine='ocr'` (line 1267)
- `@output` → `engine='output'` (line 1337)
- `@webscrape` → `engine='webscraping'` (line 1362)
- `@input` → `engine='userinput'` (line 1386)
- `@execute` → `engine='execute'` (line 1410)
- `@index` → `engine='index'` (line 1440)
- `@finetune` → `engine='finetune'` (line 1494)
- `@caption` → `engine='imagecaptioning'` (line 1518)

### 2. **Engine Resolution Process**
When a decorated function is called:
1. The decorator creates an `Argument` object with all parameters
2. Calls `EngineRepository.query(engine='engine_name', ...)`
3. `EngineRepository.get(engine_name)` is called, which:
   - Checks if engine is already registered in `_engines` dict
   - If not, dynamically imports the engine module from `symai.backend.engines.{engine_name}`
   - Registers all Engine subclasses from that module
4. The engine instance is retrieved and its processing method is called

### 3. **Key Insight: No Dynamic Engine Detection**
- There's NO automatic detection of which engine to use based on the expression
- Engine selection is hardcoded in the decorator (e.g., `@search` always uses 'search' engine)
- The `@bind` decorator is for binding to engine properties, not for routing requests

### 4. **Current BatchScheduler Limitation**
The current BatchScheduler hardcodes to 'neurosymbolic' engine because:
- Line 30: `repository.get('neurosymbolic').__setattr__("executor_callback",self.executor_callback)`
- It only intercepts calls to the neurosymbolic engine

### 5. **For Multi-Engine Support**
To support multiple engines in BatchScheduler, we need to:
1. **Register callbacks for all engines** we want to batch
2. **Detect engine from the decorator used** (not from the expression itself)
3. **Route batches to appropriate engines** based on the decorator's engine specification

The key realization is that engine selection happens at the decorator level, not at the expression level. Each decorator knows exactly which engine it needs.

## Implementation Approach

### Simple Multi-Engine BatchScheduler
1. Accept a list of engine names to support
2. Register our executor_callback with each engine
3. Track which engine each call is targeted for
4. Group calls by engine and batch them separately
5. Maintain separate queues and locks per engine

### Code Flow Example
```
User calls expression with @search decorator
→ @search creates Argument and calls EngineRepository.query(engine='search', ...)
→ EngineRepository.get('search') returns the search engine instance
→ Our executor_callback (registered on search engine) intercepts the call
→ We queue it in the 'search' engine queue
→ Batch processor groups search calls and executes them together
```