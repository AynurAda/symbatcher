"""
Async ChatGPT Batch Engine for symbolicai

This engine provides asynchronous batch processing for ChatGPT API calls
while maintaining full semantic compatibility with GPTXChatEngine.
"""

import asyncio
from typing import List, Tuple, Any, Optional
from openai import AsyncOpenAI
from symai.backend.base import BatchEngine
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine
from symai.utils import CustomUserWarning
import logging

logger = logging.getLogger(__name__)


class AsyncGPTXBatchEngine(BatchEngine):
    """
    Async batch engine that processes multiple ChatGPT requests concurrently.
    
    This engine is a drop-in replacement for GPTXChatEngine when used with
    BatchScheduler, providing significant performance improvements for batch
    processing while maintaining identical behavior for each individual request.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the async batch engine.
        
        Args:
            api_key: OpenAI API key (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
        """
        super().__init__()
        
        # Create a regular GPTXChatEngine instance for all the logic
        self._sync_engine = GPTXChatEngine(api_key, model)
        
        # Create async client
        api_key = api_key or self._sync_engine.config.get('NEUROSYMBOLIC_ENGINE_API_KEY')
        self.async_client = AsyncOpenAI(api_key=api_key)
        
        # Mark as batch-capable
        self.allows_batching = True
        
        # Copy essential attributes from sync engine
        self.config = self._sync_engine.config
        self.model = self._sync_engine.model
        self.tokenizer = self._sync_engine.tokenizer
        self.max_context_tokens = self._sync_engine.max_context_tokens
        self.max_response_tokens = self._sync_engine.max_response_tokens
        self.seed = self._sync_engine.seed
        self.name = self.__class__.__name__
    
    def forward(self, arguments: List[Any]) -> Tuple[List[Any], List[dict]]:
        """
        Synchronous interface for BatchScheduler.
        
        Processes a batch of arguments asynchronously while providing a
        synchronous interface for compatibility with BatchScheduler.
        
        Args:
            arguments: List of argument objects to process
            
        Returns:
            Tuple of (outputs, metadatas) where outputs is a list of response
            lists and metadatas is a list of metadata dicts
        """
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
        """
        Process batch asynchronously.
        
        Creates async tasks for all arguments and processes them concurrently.
        """
        # Create tasks for all arguments
        tasks = [self._process_single_async(arg) for arg in arguments]
        
        # Wait for all tasks to complete, capturing exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        outputs = []
        metadatas = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle errors gracefully - return error info
                error_msg = f"Error processing request {i}: {str(result)}"
                logger.error(error_msg)
                outputs.append([error_msg])
                metadatas.append({
                    'error': True,
                    'exception': str(result),
                    'raw_output': None
                })
            else:
                outputs.append(result[0])
                metadatas.append(result[1])
        
        return outputs, metadatas
    
    async def _process_single_async(self, argument) -> Tuple[List[str], dict]:
        """
        Process a single request asynchronously.
        
        This method reuses ALL the logic from GPTXChatEngine for preparing
        requests and processing responses, only making the actual API call
        asynchronous.
        """
        # Use sync engine for ALL preparation logic
        self._sync_engine.prepare(argument)
        
        # Get prepared messages and apply truncation
        kwargs = argument.kwargs
        
        # Check if truncate method exists (for compatibility with different symbolicai versions)
        if hasattr(self._sync_engine, 'truncate'):
            messages = self._sync_engine.truncate(
                argument.prop.prepared_input,
                kwargs.get('truncation_percentage', argument.prop.truncation_percentage),
                kwargs.get('truncation_type', argument.prop.truncation_type)
            )
        else:
            # Fall back to using prepared_input directly
            messages = argument.prop.prepared_input
        
        # Get the complete payload using sync engine logic
        if hasattr(self._sync_engine, '_prepare_request_payload'):
            payload = self._sync_engine._prepare_request_payload(messages, argument)
        else:
            # Fallback: construct payload manually
            payload = {
                'model': self.model,
                'messages': messages,
                'temperature': kwargs.get('temperature', 0.7),
                'stream': kwargs.get('stream', False)
            }
            if 'max_tokens' in kwargs:
                payload['max_tokens'] = kwargs['max_tokens']
            if 'seed' in kwargs:
                payload['seed'] = kwargs['seed']
        
        # Handle except_remedy
        except_remedy = kwargs.get('except_remedy')
        
        try:
            # Make the async API call - this is the only async part!
            res = await self.async_client.chat.completions.create(**payload)
            
        except Exception as e:
            if except_remedy is not None:
                # Handle remedy synchronously as in the original
                # Note: except_remedy expects a sync callback, so we provide a wrapper
                async def async_callback(**kwargs):
                    return await self.async_client.chat.completions.create(**kwargs)
                
                # Create a sync wrapper for the async callback
                def sync_callback(**kwargs):
                    return asyncio.run(async_callback(**kwargs))
                
                res = except_remedy(self._sync_engine, e, sync_callback, argument)
            else:
                # Re-raise the exception to be caught by gather()
                raise e
        
        # Process response using sync engine logic
        metadata = {'raw_output': res}
        if payload.get('tools') and hasattr(self._sync_engine, '_process_function_calls'):
            metadata = self._sync_engine._process_function_calls(res, metadata)
        
        # Extract output content
        output = [r.message.content for r in res.choices]
        
        return output, metadata
    
    def id(self):
        """Return engine ID - delegates to sync engine."""
        return self._sync_engine.id()
    
    def command(self, *args, **kwargs):
        """
        Handle command updates - delegates to sync engine.
        
        This ensures configuration changes are properly propagated.
        """
        self._sync_engine.command(*args, **kwargs)
        # Update our references to match
        self.model = self._sync_engine.model
        self.seed = self._sync_engine.seed
        
        # Update async client if API key changed
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            self.async_client = AsyncOpenAI(api_key=kwargs['NEUROSYMBOLIC_ENGINE_API_KEY'])
    
    def prepare(self, argument):
        """Delegate to sync engine for compatibility."""
        return self._sync_engine.prepare(argument)
    
    def compute_remaining_tokens(self, prompts: list) -> int:
        """Delegate to sync engine."""
        return self._sync_engine.compute_remaining_tokens(prompts)
    
    def compute_required_tokens(self, messages):
        """Delegate to sync engine."""
        return self._sync_engine.compute_required_tokens(messages)