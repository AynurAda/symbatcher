"""
Rate-limited Async Gemini Reasoning Batch Engine

This engine provides rate limiting capabilities for Google Gemini API with an improved API:
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
from google import genai
from google.genai import types
from aiolimiter import AsyncLimiter
from symai.backend.base import BatchEngine
from symai.backend.engines.neurosymbolic.engine_google_geminiX_reasoning import GeminiXReasoningEngine

logger = logging.getLogger(__name__)


class RateLimitedGeminiClient:
    """Minimal wrapper that adds rate limiting to Google Gemini client."""

    def __init__(self, api_key: str, tokens_per_min: int = 30000,
                 requests_per_min: int = 60, max_retries: int = 3):
        """
        Initialize rate limited client.

        Args:
            api_key: Google API key
            tokens_per_min: Maximum tokens per minute
            requests_per_min: Maximum requests per minute
            max_retries: Maximum retry attempts on rate limit errors
        """
        self.client = genai.Client(api_key=api_key)
        self.token_limiter = AsyncLimiter(tokens_per_min, 60)
        self.request_limiter = AsyncLimiter(requests_per_min, 60)
        self.max_retries = max_retries

    async def generate_content(self, model: str, contents: List, config: types.GenerateContentConfig) -> Any:
        """
        Generate content with rate limiting and retry logic.

        Args:
            model: Model name
            contents: Content list
            config: Generation config

        Returns:
            Gemini generation response
        """
        # Estimate tokens (rough estimate based on content)
        estimated_tokens = self._estimate_tokens(contents, config)

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Acquire rate limit capacity
                async with self.request_limiter:
                    # Simple token acquisition (acquire multiple times)
                    for _ in range(min(estimated_tokens, 1000)):
                        await self.token_limiter.acquire()

                    # Make API call in thread to avoid blocking
                    # Note: Google Gemini SDK doesn't have native async support yet
                    return await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model,
                        contents=contents,
                        config=config
                    )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if 'rate limit' in error_str or 'quota' in error_str or '429' in error_str:
                    if attempt < self.max_retries - 1:
                        # Calculate wait time
                        wait_time = self._get_retry_wait_time(e, attempt)
                        logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Final attempt failed - raise
                        raise
                # Check if it's a connection/timeout error
                elif 'connection' in error_str or 'timeout' in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = min(5.0 * (attempt + 1), 15.0)  # Max 15 sec wait
                        logger.warning(f"Connection error, retrying in {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        # Final attempt failed
                        raise
                else:
                    # Other errors - raise immediately
                    raise

        # Should not reach here, but just in case
        raise last_error

    def _estimate_tokens(self, contents: List, config: types.GenerateContentConfig) -> int:
        """Estimate token count for contents."""
        # Rough estimate: count characters in text parts
        total_chars = 0
        for content in contents:
            if hasattr(content, 'parts'):
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        total_chars += len(part.text)

        # Add estimated response tokens
        max_output = config.max_output_tokens or 1024

        # Rough conversion: 4 chars per token
        return (total_chars // 4) + max_output + 100

    def _get_retry_wait_time(self, error: Exception, attempt: int) -> float:
        """Calculate retry wait time."""
        # Check for Retry-After header (if available in error)
        error_str = str(error)
        if 'retry after' in error_str.lower():
            try:
                # Try to extract number from error message
                import re
                match = re.search(r'retry after (\d+)', error_str.lower())
                if match:
                    return float(match.group(1))
            except Exception:
                pass

        # Exponential backoff with jitter
        return (2 ** attempt) + random.uniform(0, 1)


class AsyncGeminiXBatchEngine(BatchEngine):
    """
    Rate-limited async batch engine for Gemini with improved API.

    Key improvement: rate_limits=None means no rate limiting (intuitive!)

    Google Gemini Rate Limits (2025):
    - Free Tier: 15 RPM, 32K TPM (Gemini 1.5 Flash), 2 RPM / 32K TPM (Pro)
    - Pay-as-you-go: Higher limits depending on model
    - Gemini 1.5 Flash: 1000 RPM, 4M TPM
    - Gemini 1.5 Pro: 360 RPM, 4M TPM

    Default limits are set conservatively for free tier.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 rate_limits: Union[None, Literal['default'], Dict[str, int]] = 'default'):
        """
        Initialize the rate-limited async batch engine.

        Args:
            api_key: Google API key (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
            rate_limits: Rate limiting configuration
                        - 'default' (default): Use model-specific defaults
                        - None: Disable rate limiting (intuitive!)
                        - Dict: Custom limits with 'tokens', 'requests', 'max_retries'
        """
        super().__init__()

        # Create sync engine for configuration and logic
        self._sync_engine = GeminiXReasoningEngine(api_key, model)

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
            self.async_client = RateLimitedGeminiClient(
                api_key,
                tokens_per_min=actual_limits.get('tokens', 30000),
                requests_per_min=actual_limits.get('requests', 60),
                max_retries=actual_limits.get('max_retries', 3)
            )
            self._has_rate_limiting = True
        else:
            # No rate limiting - use regular client
            self.async_client = genai.Client(api_key=api_key)
            self._has_rate_limiting = False

        # Mark as batch-capable
        self.allows_batching = True

        # Copy essential attributes
        self.config = self._sync_engine.config
        self.model = self._sync_engine.model
        self.tokenizer = self._sync_engine.tokenizer
        self.max_context_tokens = self._sync_engine.max_context_tokens
        self.max_response_tokens = self._sync_engine.max_response_tokens
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
        """
        Process a single request asynchronously.

        This method reuses ALL the logic from GeminiXReasoningEngine, only making
        the actual API call asynchronous.
        """
        # Use sync engine for ALL preparation logic
        self._sync_engine.prepare(argument)

        # Get kwargs and prepared input - exact same as sync engine
        kwargs = argument.kwargs
        system, prompt = argument.prop.prepared_input
        payload = self._sync_engine._prepare_request_payload(argument)
        except_remedy = kwargs.get('except_remedy')

        # Build contents - exact same as sync engine
        contents = []
        for msg in prompt:
            role = msg['role']
            parts_list = msg['content']
            contents.append(types.Content(role=role, parts=parts_list))

        try:
            # Build generation config - exact same as sync engine
            generation_config = types.GenerateContentConfig(
                max_output_tokens=payload.get('max_output_tokens'),
                temperature=payload.get('temperature', 1.0),
                top_p=payload.get('top_p', 0.95),
                top_k=payload.get('top_k', 40),
                stop_sequences=payload.get('stop_sequences'),
                response_mime_type=payload.get('response_mime_type', 'text/plain'),
            )

            if payload.get('system_instruction'):
                generation_config.system_instruction = payload['system_instruction']

            if payload.get('thinking_config'):
                generation_config.thinking_config = payload['thinking_config']

            if payload.get('tools'):
                generation_config.tools = payload['tools']
                generation_config.automatic_function_calling = payload['automatic_function_calling']

            # Make the async API call - this is the only async part!
            if self._has_rate_limiting:
                # Use rate-limited client
                res = await self.async_client.generate_content(
                    model=kwargs.get('model', self.model),
                    contents=contents,
                    config=generation_config
                )
            else:
                # Use regular client (still async via asyncio.to_thread)
                res = await asyncio.to_thread(
                    self.async_client.models.generate_content,
                    model=kwargs.get('model', self.model),
                    contents=contents,
                    config=generation_config
                )

        except Exception as e:
            # Handle API key errors - exact same as sync engine
            if self.async_client.api_key is None or self.async_client.api_key == '':
                msg = 'Google API key is not set. Please set it in the config file or pass it as an argument to the command method.'
                logger.error(msg)
                if self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] is None or self.config['NEUROSYMBOLIC_ENGINE_API_KEY'] == '':
                    from symai.utils import CustomUserWarning
                    CustomUserWarning(msg, raise_with=ValueError)
                api_key = self.config['NEUROSYMBOLIC_ENGINE_API_KEY']
                if self._has_rate_limiting:
                    limits = self._get_default_limits(self.model)
                    self.async_client = RateLimitedGeminiClient(
                        api_key,
                        tokens_per_min=limits.get('tokens', 30000),
                        requests_per_min=limits.get('requests', 60),
                        max_retries=limits.get('max_retries', 3)
                    )
                else:
                    self.async_client = genai.Client(api_key=api_key)

            if except_remedy is not None:
                # Handle remedy - create async wrapper
                async def async_callback(**kwargs):
                    if self._has_rate_limiting:
                        return await self.async_client.generate_content(**kwargs)
                    else:
                        return await asyncio.to_thread(
                            self.async_client.models.generate_content,
                            **kwargs
                        )

                # Create a sync wrapper for the async callback
                def sync_callback(**kwargs):
                    return asyncio.run(async_callback(**kwargs))

                res = except_remedy(self._sync_engine, e, sync_callback, argument)
            else:
                # Re-raise the exception to be caught by gather()
                raise e

        # Process response using sync engine logic - exact same as sync engine
        metadata = {'raw_output': res}
        if payload.get('tools'):
            metadata = self._sync_engine._process_function_calls(res, metadata)

        if kwargs.get('raw_output', False):
            return [res], metadata

        output = self._sync_engine._collect_response(res)

        if output['thinking']:
            metadata['thinking'] = output['thinking']

        processed_text = output['text']
        if argument.prop.response_format:
            # Safely remove JSON markdown formatting if present
            processed_text = processed_text.replace('```json', '').replace('```', '')

        return [processed_text], metadata

    def _get_default_limits(self, model: Optional[str]) -> Optional[Dict[str, int]]:
        """Get default rate limits for a model."""
        # Check if rate limiting is explicitly disabled via env
        if os.getenv('SYMBATCHER_RATE_LIMITING_ENABLED', '').lower() == 'false':
            return None

        # Check environment variables
        env_tokens = os.getenv('SYMBATCHER_GEMINI_TOKENS_PER_MIN')
        env_requests = os.getenv('SYMBATCHER_GEMINI_REQUESTS_PER_MIN')

        if env_tokens or env_requests:
            return {
                'tokens': int(env_tokens) if env_tokens else 30000,
                'requests': int(env_requests) if env_requests else 60,
                'max_retries': int(os.getenv('SYMBATCHER_GEMINI_MAX_RETRIES', '3'))
            }

        # Model defaults - Conservative free tier limits
        # Using 90% of actual limits to provide safety margin
        limits = {
            'gemini-1.5-flash': {'tokens': 28800, 'requests': 13},  # 90% of 32K TPM, 15 RPM (free)
            'gemini-1.5-flash-8b': {'tokens': 28800, 'requests': 13},
            'gemini-1.5-pro': {'tokens': 28800, 'requests': 1},  # 90% of 32K TPM, 2 RPM (free)
            'gemini-2.0-flash': {'tokens': 28800, 'requests': 13},
            'gemini-2.0-flash-exp': {'tokens': 28800, 'requests': 13},
        }

        # Check if model matches any pattern
        if model:
            for model_pattern, model_limits in limits.items():
                if model.startswith(model_pattern):
                    model_limits['max_retries'] = 3
                    return model_limits

        # Default conservative limits for unknown models
        return {'tokens': 28800, 'requests': 13, 'max_retries': 3}

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

        # Update async client if API key changed
        if 'NEUROSYMBOLIC_ENGINE_API_KEY' in kwargs:
            api_key = kwargs['NEUROSYMBOLIC_ENGINE_API_KEY']
            if self._has_rate_limiting:
                # Get current limits
                limits = self._get_default_limits(self.model)
                self.async_client = RateLimitedGeminiClient(
                    api_key,
                    tokens_per_min=limits.get('tokens', 30000),
                    requests_per_min=limits.get('requests', 60),
                    max_retries=limits.get('max_retries', 3)
                )
            else:
                self.async_client = genai.Client(api_key=api_key)

    def prepare(self, argument):
        """Delegate to sync engine for compatibility."""
        return self._sync_engine.prepare(argument)

    def compute_remaining_tokens(self, prompts: list) -> int:
        """Delegate to sync engine."""
        return self._sync_engine.compute_remaining_tokens(prompts)

    def compute_required_tokens(self, messages):
        """Delegate to sync engine."""
        return self._sync_engine.compute_required_tokens(messages)
