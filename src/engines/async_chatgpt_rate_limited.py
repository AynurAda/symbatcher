"""
Rate-limited Async ChatGPT Batch Engine

This engine provides rate limiting capabilities with an improved API:
- rate_limits=None means no rate limiting (intuitive!)
- rate_limits='default' means use model defaults
- rate_limits={...} means use custom limits
"""

import asyncio
import os
import time
import random
import logging
from typing import List, Tuple, Any, Optional, Dict, Union, Literal
from openai import AsyncOpenAI
import openai
from aiolimiter import AsyncLimiter
from symai.backend.base import BatchEngine
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine

logger = logging.getLogger(__name__)


class RateLimitedClient:
    """Minimal wrapper that adds rate limiting to AsyncOpenAI client."""
    
    def __init__(self, api_key: str, tokens_per_min: int = 90000, 
                 requests_per_min: int = 3500, max_retries: int = 3):
        """
        Initialize rate limited client.
        
        Args:
            api_key: OpenAI API key
            tokens_per_min: Maximum tokens per minute
            requests_per_min: Maximum requests per minute
            max_retries: Maximum retry attempts on rate limit errors
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.token_limiter = AsyncLimiter(tokens_per_min, 60)
        self.request_limiter = AsyncLimiter(requests_per_min, 60)
        self.max_retries = max_retries
        self._token_encoding = None
    
    async def create_completion(self, messages: List[Dict], **kwargs) -> Any:
        """
        Create completion with rate limiting and retry logic.
        
        Args:
            messages: Chat messages
            **kwargs: Additional arguments for OpenAI API
            
        Returns:
            OpenAI completion response
        """
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(messages, kwargs.get('model', 'gpt-4'))
        estimated_tokens += kwargs.get('max_tokens', 500)
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Acquire rate limit capacity
                async with self.request_limiter:
                    # Simple token acquisition (acquire multiple times)
                    for _ in range(min(estimated_tokens, 1000)):
                        await self.token_limiter.acquire()
                    
                    # Make API call
                    return await self.client.chat.completions.create(
                        messages=messages, **kwargs
                    )
                    
            except openai.RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Calculate wait time
                    wait_time = self._get_retry_wait_time(e, attempt)
                    logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed - return error as response
                    return self._format_error_response(e)
        
        return self._format_error_response(last_error)
    
    def _estimate_tokens(self, messages: List[Dict], model: str) -> int:
        """Estimate token count for messages."""
        if self._token_encoding is None:
            try:
                import tiktoken
                self._token_encoding = tiktoken.encoding_for_model(model)
            except Exception:
                pass
        
        if self._token_encoding:
            text = ""
            for msg in messages:
                text += f"{msg.get('role', '')}: {msg.get('content', '')}\n"
            return len(self._token_encoding.encode(text))
        else:
            # Rough estimate
            text = str(messages)
            return len(text) // 4 + 100
    
    def _get_retry_wait_time(self, error: openai.RateLimitError, attempt: int) -> float:
        """Calculate retry wait time."""
        # Check for Retry-After header
        if hasattr(error, 'response') and hasattr(error.response, 'headers'):
            retry_after = error.response.headers.get('retry-after')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        
        # Exponential backoff with jitter
        return (2 ** attempt) + random.uniform(0, 1)
    
    def _format_error_response(self, error: Exception):
        """Format error to match expected response format."""
        class MockResponse:
            def __init__(self, error_msg):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': error_msg})()
                })()]
        
        return MockResponse(f"Rate limit error: {str(error)}")


class AsyncGPTXBatchEngine(BatchEngine):
    """
    Rate-limited async batch engine with improved API.
    
    Key improvement: rate_limits=None means no rate limiting (intuitive!)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 rate_limits: Union[None, Literal['default'], Dict[str, int]] = 'default'):
        """
        Initialize the rate-limited async batch engine.
        
        Args:
            api_key: OpenAI API key (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
            rate_limits: Rate limiting configuration
                        - 'default' (default): Use model-specific defaults
                        - None: Disable rate limiting (intuitive!)
                        - Dict: Custom limits with 'tokens', 'requests', 'max_retries'
        """
        super().__init__()
        
        # Create sync engine for configuration and logic
        self._sync_engine = GPTXChatEngine(api_key, model)
        
        # Get API key
        api_key = api_key or self._sync_engine.config.get('NEUROSYMBOLIC_ENGINE_API_KEY')
        
        # Handle rate limits with clearer logic
        actual_limits = None
        if rate_limits is None:
            # Explicitly disabled - this is the intuitive behavior!
            logger.info("Rate limiting explicitly disabled")
            actual_limits = None
        elif rate_limits == 'default':
            # Use model defaults
            actual_limits = self._get_default_limits(self._sync_engine.model)
            if actual_limits:
                logger.info(f"Using default rate limits for {self._sync_engine.model}: {actual_limits}")
            else:
                logger.info(f"No default rate limits for {self._sync_engine.model}, rate limiting disabled")
        elif isinstance(rate_limits, dict):
            # Custom limits
            actual_limits = rate_limits
            logger.info(f"Using custom rate limits: {actual_limits}")
        else:
            raise ValueError(f"Invalid rate_limits value: {rate_limits}. Must be 'default', None, or a dict")
        
        # Create client based on whether we have rate limits
        if actual_limits:
            self.async_client = RateLimitedClient(
                api_key,
                tokens_per_min=actual_limits.get('tokens', 90000),
                requests_per_min=actual_limits.get('requests', 3500),
                max_retries=actual_limits.get('max_retries', 3)
            )
            self._has_rate_limiting = True
        else:
            # No rate limiting - use regular client
            self.async_client = AsyncOpenAI(api_key=api_key)
            self._has_rate_limiting = False
        
        # Mark as batch-capable
        self.allows_batching = True
        
        # Copy essential attributes
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
        
        Args:
            arguments: List of argument objects to process
            
        Returns:
            Tuple of (outputs, metadatas)
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(self._forward_async(arguments), loop)
            return future.result()
        except RuntimeError:
            return asyncio.run(self._forward_async(arguments))
    
    async def _forward_async(self, arguments: List[Any]) -> Tuple[List[Any], List[dict]]:
        """Process batch asynchronously."""
        tasks = [self._process_single_async(arg) for arg in arguments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        outputs = []
        metadatas = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Error processing request {i}: {str(result)}"
                logger.error(f"{error_msg} (Type: {type(result).__name__})")
                # Include more detailed error info in metadata
                import traceback
                metadatas.append({
                    'error': True,
                    'exception': str(result),
                    'exception_type': type(result).__name__,
                    'traceback': traceback.format_exception(type(result), result, result.__traceback__),
                    'raw_output': None
                })
                outputs.append([error_msg])
            else:
                outputs.append(result[0])
                metadatas.append(result[1])
        
        return outputs, metadatas
    
    async def _process_single_async(self, argument) -> Tuple[List[str], dict]:
        """Process a single request asynchronously."""
        try:
            # Use sync engine for preparation
            self._sync_engine.prepare(argument)
            
            # Get messages
            messages = self._get_messages(argument)
            
            # Prepare payload
            payload = self._prepare_payload(messages, argument)
            
            # Make API call
            if self._has_rate_limiting:
                # Use rate-limited client
                res = await self.async_client.create_completion(**payload)
            else:
                # Use regular client
                res = await self.async_client.chat.completions.create(**payload)
            
            # Process response
            metadata = {'raw_output': res}
            if payload.get('tools') and hasattr(self._sync_engine, '_process_function_calls'):
                metadata = self._sync_engine._process_function_calls(res, metadata)
            
            # Extract output
            output = [r.message.content for r in res.choices]
            
            return output, metadata
            
        except openai.APIConnectionError as e:
            logger.warning(f"OpenAI API connection error: {e}")
            raise
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit error: {e}")
            raise
        except openai.APIError as e:
            logger.warning(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _process_single_async: {type(e).__name__}: {e}")
            raise
    
    def _get_messages(self, argument) -> List[Dict]:
        """Get messages from argument."""
        kwargs = argument.kwargs
        
        # Check if truncate method exists
        if hasattr(self._sync_engine, 'truncate'):
            messages = self._sync_engine.truncate(
                argument.prop.prepared_input,
                kwargs.get('truncation_percentage', argument.prop.truncation_percentage),
                kwargs.get('truncation_type', argument.prop.truncation_type)
            )
        else:
            messages = argument.prop.prepared_input
        
        return messages
    
    def _prepare_payload(self, messages: List[Dict], argument) -> Dict:
        """Prepare API request payload."""
        kwargs = argument.kwargs
        
        if hasattr(self._sync_engine, '_prepare_request_payload'):
            payload = self._sync_engine._prepare_request_payload(messages, argument)
        else:
            # Fallback
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
        
        # Handle deprecated max_tokens -> max_completion_tokens conversion
        if 'max_tokens' in payload and 'max_completion_tokens' not in payload:
            # Use max_completion_tokens for newer API compatibility
            payload['max_completion_tokens'] = payload.pop('max_tokens')
            logger.debug(f"Converted max_tokens to max_completion_tokens: {payload['max_completion_tokens']}")
        
        return payload
    
    def _get_default_limits(self, model: Optional[str]) -> Optional[Dict[str, int]]:
        """Get default rate limits for a model."""
        # Check if rate limiting is explicitly disabled via env
        if os.getenv('SYMBATCHER_RATE_LIMITING_ENABLED', '').lower() == 'false':
            return None
        
        # Check environment variables
        env_tokens = os.getenv('SYMBATCHER_OPENAI_TOKENS_PER_MIN')
        env_requests = os.getenv('SYMBATCHER_OPENAI_REQUESTS_PER_MIN')
        
        if env_tokens or env_requests:
            return {
                'tokens': int(env_tokens) if env_tokens else 90000,
                'requests': int(env_requests) if env_requests else 3500,
                'max_retries': int(os.getenv('SYMBATCHER_OPENAI_MAX_RETRIES', '3'))
            }
        
        # Model defaults (90% of actual limits)
        limits = {
            'gpt-4': {'tokens': 81000, 'requests': 3150},
            'gpt-4o': {'tokens': 135000, 'requests': 4500},
            'gpt-4-turbo': {'tokens': 135000, 'requests': 4500},
            'gpt-4-turbo-preview': {'tokens': 135000, 'requests': 4500},
            'gpt-3.5-turbo': {'tokens': 180000, 'requests': 9000},
            'gpt-3.5-turbo-16k': {'tokens': 180000, 'requests': 9000},
        }
        
        if model in limits:
            limits[model]['max_retries'] = 3
            return limits[model]
        
        return None
    
    def id(self):
        """Return engine ID."""
        return self._sync_engine.id()
    
    def command(self, *args, **kwargs):
        """Handle command updates."""
        self._sync_engine.command(*args, **kwargs)
        self.model = self._sync_engine.model
        self.seed = self._sync_engine.seed
        
        # Update client if API key changed
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
            if self._has_rate_limiting:
                # Get current limits
                limits = self._get_default_limits(self.model)
                self.async_client = RateLimitedClient(
                    api_key,
                    tokens_per_min=limits.get('tokens', 90000),
                    requests_per_min=limits.get('requests', 3500),
                    max_retries=limits.get('max_retries', 3)
                )
            else:
                self.async_client = AsyncOpenAI(api_key=api_key)
    
    def prepare(self, argument):
        """Delegate to sync engine."""
        return self._sync_engine.prepare(argument)
    
    def compute_remaining_tokens(self, prompts: list) -> int:
        """Delegate to sync engine."""
        return self._sync_engine.compute_remaining_tokens(prompts)
    
    def compute_required_tokens(self, messages):
        """Delegate to sync engine."""
        return self._sync_engine.compute_required_tokens(messages)