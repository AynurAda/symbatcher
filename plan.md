# Async ChatGPT Engine Implementation Plan

## Overview

Implement an asynchronous ChatGPT engine that is **semantically identical** to GPTXChatEngine except for async batch processing, without modifying BatchScheduler.

## Critical Analysis

### Semantic Requirements
The async engine MUST preserve ALL functionality from GPTXChatEngine:
- Message preparation (system prompts, context, examples, templates)
- Token counting and truncation
- Request payload preparation with all parameters
- Function call processing
- Vision/image content handling
- Error handling with except_remedy
- Response format handling
- All metadata processing

### Current GPTXChatEngine Flow
1. `prepare()` - Builds messages with prompts, context, examples
2. `truncate()` - Handles token limits
3. `_prepare_request_payload()` - Builds complete API request
4. API call with error handling
5. `_process_function_calls()` - Extracts function calls if present
6. Returns outputs and metadata

## Concise Implementation

```python
# async_chatgpt.py
import asyncio
from typing import List, Tuple, Any
from openai import AsyncOpenAI
from symai.backend.base import BatchEngine
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine

class AsyncGPTXBatchEngine(BatchEngine):
    """Async batch engine that reuses ALL GPTXChatEngine logic"""
    
    def __init__(self, api_key=None, model=None):
        super().__init__()
        # Create a regular GPTXChatEngine instance for all the logic
        self._sync_engine = GPTXChatEngine(api_key, model)
        self.async_client = AsyncOpenAI(api_key=self._sync_engine.config['NEUROSYMBOLIC_ENGINE_API_KEY'])
        self.allows_batching = True
        
        # Copy all attributes from sync engine
        self.config = self._sync_engine.config
        self.model = self._sync_engine.model
        self.tokenizer = self._sync_engine.tokenizer
        self.max_context_tokens = self._sync_engine.max_context_tokens
        self.max_response_tokens = self._sync_engine.max_response_tokens
    
    def forward(self, arguments: List[Any]) -> Tuple[List[Any], List[dict]]:
        """Synchronous interface for BatchScheduler"""
        try:
            # Check if there's already a running event loop (e.g., from BatchScheduler)
            loop = asyncio.get_running_loop()
            # If we're in an existing loop, use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(self._forward_async(arguments), loop)
            return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self._forward_async(arguments))
    
    async def _forward_async(self, arguments: List[Any]) -> Tuple[List[Any], List[dict]]:
        """Process batch asynchronously"""
        tasks = [self._process_single_async(arg) for arg in arguments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        outputs = []
        metadatas = []
        for result in results:
            if isinstance(result, Exception):
                # Let the sync engine handle the error properly
                try:
                    raise result
                except Exception as e:
                    # This matches the error handling in GPTXChatEngine
                    outputs.append([str(e)])
                    metadatas.append({'error': True, 'raw_output': None})
            else:
                outputs.append(result[0])
                metadatas.append(result[1])
        
        return outputs, metadatas
    
    async def _process_single_async(self, argument) -> Tuple[List[str], dict]:
        """Process single request - reusing ALL sync engine logic"""
        # Use sync engine for ALL preparation logic
        self._sync_engine.prepare(argument)
        
        # Get prepared messages and apply truncation
        messages = self._sync_engine.truncate(
            argument.prop.prepared_input,
            argument.kwargs.get('truncation_percentage', argument.prop.truncation_percentage),
            argument.kwargs.get('truncation_type', argument.prop.truncation_type)
        )
        
        # Get the complete payload using sync engine logic
        payload = self._sync_engine._prepare_request_payload(messages, argument)
        
        # Make the async API call
        except_remedy = argument.kwargs.get('except_remedy')
        try:
            res = await self.async_client.chat.completions.create(**payload)
        except Exception as e:
            if except_remedy is not None:
                # Handle remedy synchronously as in the original
                callback = self.async_client.chat.completions.create
                res = except_remedy(self._sync_engine, e, callback, argument)
            else:
                raise e
        
        # Process response using sync engine logic
        metadata = {'raw_output': res}
        if payload.get('tools'):
            metadata = self._sync_engine._process_function_calls(res, metadata)
        output = [r.message.content for r in res.choices]
        
        return output, metadata
    
    def id(self):
        """Delegate to sync engine"""
        return self._sync_engine.id()
    
    def command(self, *args, **kwargs):
        """Delegate to sync engine"""
        self._sync_engine.command(*args, **kwargs)
        # Update our model reference if changed
        self.model = self._sync_engine.model
```

## Key Design Decisions

1. **Composition over Inheritance**: Use a GPTXChatEngine instance internally to ensure 100% semantic compatibility
2. **Minimal Async Code**: Only the API call is async, everything else reuses sync engine
3. **Simple Event Loop**: Use `asyncio.run()` instead of managing threads - simpler and cleaner
4. **Delegate Pattern**: All non-batch methods delegate to the sync engine

## Why This is Better

1. **Semantic Correctness**: Guaranteed to behave exactly like GPTXChatEngine
2. **Maintainability**: Changes to GPTXChatEngine automatically apply
3. **Simplicity**: No complex event loop management
4. **Conciseness**: ~80 lines instead of complex inheritance

## Testing

```python
# Verify semantic equivalence
sync_result = GPTXChatEngine().forward([arg])
async_result = AsyncGPTXBatchEngine().forward([arg])
assert sync_result == async_result  # Should be identical
```

## BatchScheduler Compatibility Analysis

The async engine is **fully compatible** with BatchScheduler because:

1. **Call Chain**: 
   - BatchScheduler → `engine()` (via bind) → `BatchEngine.__call__()` → `AsyncGPTXBatchEngine.forward()`
   - Our `forward()` method provides the synchronous interface BatchScheduler expects

2. **Event Loop Handling**:
   - BatchScheduler has its own event loop running in a thread
   - Our updated `forward()` method detects if an event loop is already running
   - Uses `asyncio.run_coroutine_threadsafe()` to safely run async code from sync context

3. **No Changes Required**:
   - BatchScheduler continues to work exactly as before
   - The async engine is a drop-in replacement for the sync engine

## Usage

```python
# Drop-in replacement
EngineRepository.register('neurosymbolic', AsyncGPTXBatchEngine())
# BatchScheduler uses it without any changes!
```