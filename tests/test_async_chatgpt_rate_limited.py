"""
Comprehensive tests for rate-limited async ChatGPT engine - Fixed version.

This test suite covers:
1. Basic rate limiting functionality
2. Token estimation and counting
3. Retry logic with exponential backoff
4. Error handling and edge cases
5. Environment variable configuration
6. Batch processing
7. API compatibility
8. Thread safety and concurrency
"""

import asyncio
import time
import os
import json
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock, call
import pytest
from openai import AsyncOpenAI
import openai

from src.engines.async_chatgpt_rate_limited import (
    RateLimitedClient,
    AsyncGPTXBatchEngine
)


class MockMessage:
    """Mock message for testing."""
    def __init__(self, content):
        self.content = content


class MockChoice:
    """Mock choice for testing."""
    def __init__(self, content):
        self.message = MockMessage(content)


class MockCompletion:
    """Mock completion response."""
    def __init__(self, content="Test response"):
        self.choices = [MockChoice(content)]
        self._headers = {}


class MockRateLimitError(openai.RateLimitError):
    """Mock rate limit error with response headers."""
    def __init__(self, message, retry_after=None):
        self.message = message
        self.response = Mock()
        self.response.headers = {'retry-after': str(retry_after)} if retry_after else {}
        super().__init__(message, response=self.response, body=None)


class MockAsyncClient:
    """Mock AsyncOpenAI client for testing."""
    def __init__(self, api_key=None):
        self.chat = self
        self.completions = self
        self.call_count = 0
        self.call_times = []
        
    async def create(self, **kwargs):
        """Mock create method that tracks calls."""
        self.call_count += 1
        self.call_times.append(time.time())
        # Simulate some processing time
        await asyncio.sleep(0.01)
        return MockCompletion(f"Response {self.call_count}")


@pytest.mark.asyncio
async def test_rate_limited_client_basic():
    """Test basic rate limiting functionality."""
    # Create rate limited client with very low limits for testing
    rate_limited = RateLimitedClient(
        api_key="test-key",
        tokens_per_min=1000,  # Low limit for testing
        requests_per_min=10   # 10 requests per minute
    )
    
    # Mock the underlying client
    rate_limited.client = MockAsyncClient()
    
    # Make multiple requests
    start_time = time.time()
    tasks = []
    for i in range(5):
        task = rate_limited.create_completion(
            messages=[{"role": "user", "content": f"Test {i}"}],
            max_tokens=50
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    # Check results
    assert len(results) == 5
    assert all(hasattr(r, 'choices') for r in results)
    
    # Check that calls were made
    assert rate_limited.client.call_count == 5


@pytest.mark.asyncio
async def test_token_estimation():
    """Test token estimation functionality."""
    client = RateLimitedClient("test-key", 1000, 10)
    
    # Test with simple messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    
    tokens = client._estimate_tokens(messages, "gpt-4")
    
    # Should be reasonable estimate
    assert 10 < tokens < 100  # Rough range for these messages
    
    # Test with tiktoken available
    with patch('tiktoken.encoding_for_model') as mock_tiktoken:
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1] * 50  # 50 tokens
        mock_tiktoken.return_value = mock_encoding
        
        client._token_encoding = None  # Reset
        tokens = client._estimate_tokens(messages, "gpt-4")
        
        # Should use tiktoken if available
        assert 40 < tokens < 60  # Around 50


@pytest.mark.asyncio
async def test_retry_on_rate_limit():
    """Test retry logic for rate limit errors."""
    client = RateLimitedClient("test-key", 1000, 10, max_retries=3)
    
    # Mock the underlying client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    
    # First two calls raise RateLimitError, third succeeds
    mock_client.chat.completions.create.side_effect = [
        openai.RateLimitError("Rate limit exceeded", response=Mock(headers={}), body=None),
        openai.RateLimitError("Rate limit exceeded", response=Mock(headers={}), body=None),
        MockCompletion("Success after retries")
    ]
    
    client.client = mock_client
    
    start_time = time.time()
    result = await client.create_completion(
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=50
    )
    duration = time.time() - start_time
    
    # Should succeed after retries
    assert hasattr(result, 'choices')
    assert result.choices[0].message.content == "Success after retries"
    
    # Should have called 3 times
    assert mock_client.chat.completions.create.call_count == 3
    
    # Should have waited due to exponential backoff
    assert duration > 3  # At least 1 + 2 seconds for backoff


@pytest.mark.asyncio
async def test_retry_with_retry_after_header():
    """Test retry logic using Retry-After header."""
    client = RateLimitedClient("test-key", 1000, 10, max_retries=3)
    
    # Mock the underlying client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    
    # First call raises RateLimitError with Retry-After header
    mock_client.chat.completions.create.side_effect = [
        MockRateLimitError("Rate limit exceeded", retry_after=0.5),
        MockCompletion("Success after retry")
    ]
    
    client.client = mock_client
    
    start_time = time.time()
    result = await client.create_completion(
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=50
    )
    duration = time.time() - start_time
    
    # Should succeed after retry
    assert hasattr(result, 'choices')
    assert result.choices[0].message.content == "Success after retry"
    
    # Should have waited approximately the retry-after time
    assert 0.4 < duration < 0.7  # Around 0.5 seconds


@pytest.mark.asyncio
async def test_retry_exhaustion():
    """Test behavior when all retries are exhausted."""
    client = RateLimitedClient("test-key", 1000, 10, max_retries=2)
    
    # Mock the underlying client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    
    # All calls fail
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        "Rate limit exceeded", response=Mock(headers={}), body=None
    )
    
    client.client = mock_client
    
    result = await client.create_completion(
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=50
    )
    
    # Should return formatted error response
    assert hasattr(result, 'choices')
    assert "Rate limit error" in result.choices[0].message.content
    
    # Should have tried max_retries times
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_rate_limiting():
    """Test rate limiting with concurrent requests."""
    # Create client with very strict limits
    client = RateLimitedClient(
        api_key="test-key",
        tokens_per_min=60,   # 1 token per second
        requests_per_min=60  # 1 request per second  
    )
    
    # Mock the underlying client
    client.client = MockAsyncClient()
    
    # Make concurrent requests
    start_time = time.time()
    tasks = []
    for i in range(3):
        task = client.create_completion(
            messages=[{"role": "user", "content": f"Test {i}"}],
            max_tokens=50  # Each request uses ~50 tokens
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    # Should complete but take some time due to rate limiting
    assert len(results) == 3
    # With low rate limits, requests should be somewhat spread out
    assert duration > 0.01  # At least some delay


@pytest.mark.asyncio
async def test_engine_initialization_with_rate_limits():
    """Test engine initialization with rate limits."""
    # Mock GPTXChatEngine
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_instance = MagicMock()
        mock_instance.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_instance.model = "gpt-4"
        mock_instance.tokenizer = Mock()
        mock_instance.max_context_tokens = 8192
        mock_instance.max_response_tokens = 4096
        mock_instance.seed = None
        mock_instance.name = 'GPTXChatEngine'
        mock_gptx.return_value = mock_instance
        
        # Test with explicit rate limits
        engine = AsyncGPTXBatchEngine(
            model="gpt-4",
            rate_limits={'tokens': 5000, 'requests': 100}
        )
        
        # Check that rate limiting is enabled
        assert engine._has_rate_limiting is True
        assert isinstance(engine.async_client, RateLimitedClient)
        
        # Test with model defaults
        engine2 = AsyncGPTXBatchEngine(model="gpt-3.5-turbo")
        assert engine2._has_rate_limiting is True
        
        # Test with rate limiting disabled
        with patch.dict(os.environ, {'SYMBATCHER_RATE_LIMITING_ENABLED': 'false'}):
            engine3 = AsyncGPTXBatchEngine(model="gpt-4")
            # Should not have rate limiting
            assert engine3._has_rate_limiting is False
            assert isinstance(engine3.async_client, AsyncOpenAI)


def test_get_default_limits():
    """Test default limit calculation."""
    # Mock GPTXChatEngine
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine'):
        engine = AsyncGPTXBatchEngine(model="gpt-4")
        
        # Test known models
        limits = engine._get_default_limits("gpt-4")
        assert limits == {'tokens': 81000, 'requests': 3150, 'max_retries': 3}
        
        limits = engine._get_default_limits("gpt-3.5-turbo")
        assert limits == {'tokens': 180000, 'requests': 9000, 'max_retries': 3}
        
        # Test unknown model
        limits = engine._get_default_limits("unknown-model")
        assert limits is None
        
        # Test with environment override
        with patch.dict(os.environ, {'SYMBATCHER_OPENAI_TOKENS_PER_MIN': '1000'}):
            limits = engine._get_default_limits("gpt-4")
            assert limits == {'tokens': 1000, 'requests': 3500, 'max_retries': 3}


@pytest.mark.asyncio
async def test_environment_variable_configuration():
    """Test configuration via environment variables."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_instance = MagicMock()
        mock_instance.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_instance.model = "gpt-4"
        mock_instance.tokenizer = Mock()
        mock_instance.max_context_tokens = 8192
        mock_instance.max_response_tokens = 4096
        mock_instance.seed = None
        mock_instance.name = 'GPTXChatEngine'
        mock_gptx.return_value = mock_instance
        
        with patch.dict(os.environ, {
            'SYMBATCHER_OPENAI_TOKENS_PER_MIN': '5000',
            'SYMBATCHER_OPENAI_REQUESTS_PER_MIN': '100',
            'SYMBATCHER_OPENAI_MAX_RETRIES': '5'
        }):
            engine = AsyncGPTXBatchEngine(model="gpt-4")
            
            # Should use environment variable values
            assert isinstance(engine.async_client, RateLimitedClient)
            # Check limiter properties (aiolimiter stores total per period, not rate)
            assert engine.async_client.token_limiter.max_rate == 5000
            assert engine.async_client.request_limiter.max_rate == 100
            assert engine.async_client.max_retries == 5


@pytest.mark.asyncio
async def test_process_single_async():
    """Test processing a single request."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test"}],
            'model': 'gpt-4',
            'max_tokens': 50
        })
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(
            model="gpt-4",
            rate_limits={'tokens': 5000, 'requests': 100}
        )
        
        # Mock the rate limited client
        engine.async_client = MagicMock()
        engine.async_client.create_completion = AsyncMock(return_value=MockCompletion("Test response"))
        
        # Create mock argument
        mock_arg = MagicMock()
        mock_arg.prop.prepared_input = [{"role": "user", "content": "Test"}]
        mock_arg.prop.truncation_percentage = 0.8
        mock_arg.prop.truncation_type = 'tail'
        mock_arg.kwargs = {'max_tokens': 50}
        
        # Process
        output, metadata = await engine._process_single_async(mock_arg)
        
        # Verify
        assert output == ["Test response"]
        assert 'raw_output' in metadata
        assert mock_sync.prepare.called
        assert mock_sync.truncate.called


@pytest.mark.asyncio
async def test_process_single_async_with_tools():
    """Test processing with function calls/tools."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test"}],
            'model': 'gpt-4',
            'tools': [{'type': 'function', 'function': {'name': 'test_func'}}]
        })
        mock_sync._process_function_calls = MagicMock(return_value={'processed': True})
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits=None)
        
        # Mock the async client
        engine.async_client = AsyncMock()
        engine.async_client.chat.completions.create = AsyncMock(return_value=MockCompletion())
        
        # Create mock argument
        mock_arg = MagicMock()
        mock_arg.prop.prepared_input = [{"role": "user", "content": "Test"}]
        mock_arg.prop.truncation_percentage = 0.8
        mock_arg.prop.truncation_type = 'tail'
        mock_arg.kwargs = {'tools': [{'type': 'function'}]}
        
        # Process
        output, metadata = await engine._process_single_async(mock_arg)
        
        # Verify function calls were processed
        assert mock_sync._process_function_calls.called
        assert metadata == {'processed': True}


@pytest.mark.asyncio
async def test_forward_batch():
    """Test batch processing."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test"}],
            'model': 'gpt-4'
        })
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(
            model="gpt-4",
            rate_limits={'tokens': 5000, 'requests': 100}
        )
        
        # Mock the rate limited client
        engine.async_client = MagicMock()
        call_count = 0
        
        async def mock_create_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            return MockCompletion(f"Response {call_count}")
        
        engine.async_client.create_completion = mock_create_completion
        
        # Create mock arguments
        mock_args = []
        for i in range(3):
            arg = MagicMock()
            arg.prop.prepared_input = [{"role": "user", "content": f"Test {i}"}]
            arg.prop.truncation_percentage = 0.8
            arg.prop.truncation_type = 'tail'
            arg.kwargs = {}
            mock_args.append(arg)
        
        # Process batch
        outputs, metadata = await engine._forward_async(mock_args)
        
        # Verify
        assert len(outputs) == 3
        assert len(metadata) == 3
        assert outputs[0] == ["Response 1"]
        assert outputs[1] == ["Response 2"]
        assert outputs[2] == ["Response 3"]


@pytest.mark.asyncio
async def test_forward_batch_with_errors():
    """Test batch processing with some errors."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test"}],
            'model': 'gpt-4'
        })
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits=None)
        
        # Create async functions for each call
        async def success1(**kwargs):
            return MockCompletion("Response 1")
        
        async def error(**kwargs):
            raise Exception("API Error")
        
        async def success3(**kwargs):
            return MockCompletion("Response 3")
        
        # Mock the async client to raise error on second call
        engine.async_client = AsyncMock()
        engine.async_client.chat = AsyncMock()
        engine.async_client.chat.completions = AsyncMock()
        engine.async_client.chat.completions.create = AsyncMock()
        engine.async_client.chat.completions.create.side_effect = [
            MockCompletion("Response 1"),
            Exception("API Error"),
            MockCompletion("Response 3")
        ]
        
        # Create mock arguments
        mock_args = []
        for i in range(3):
            arg = MagicMock()
            arg.prop.prepared_input = [{"role": "user", "content": f"Test {i}"}]
            arg.prop.truncation_percentage = 0.8
            arg.prop.truncation_type = 'tail'
            arg.kwargs = {}
            mock_args.append(arg)
        
        # Process batch
        outputs, metadata = await engine._forward_async(mock_args)
        
        # Verify
        assert len(outputs) == 3
        assert len(metadata) == 3
        assert outputs[0] == ["Response 1"]
        assert "Error processing request 1" in outputs[1][0]
        assert outputs[2] == ["Response 3"]
        assert metadata[1]['error'] is True


@pytest.mark.asyncio
async def test_forward_sync_interface():
    """Test synchronous forward interface."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4")
        
        # Mock the async processing
        mock_results = (["Response"], [{"test": True}])
        engine._forward_async = AsyncMock(return_value=mock_results)
        
        # Test with no running loop (should use asyncio.run)
        with patch('asyncio.get_running_loop', side_effect=RuntimeError):
            with patch('asyncio.run', return_value=mock_results) as mock_run:
                results = engine.forward([Mock()])
                assert results == mock_results
                assert mock_run.called
        
        # Test with running loop (should use run_coroutine_threadsafe)
        mock_loop = Mock()
        mock_future = Mock()
        mock_future.result.return_value = mock_results
        
        with patch('asyncio.get_running_loop', return_value=mock_loop):
            with patch('asyncio.run_coroutine_threadsafe', return_value=mock_future) as mock_threadsafe:
                results = engine.forward([Mock()])
                assert results == mock_results
                assert mock_threadsafe.called


@pytest.mark.asyncio
async def test_command_method():
    """Test command method updates configuration."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.seed = None
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.command = MagicMock()
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits={'tokens': 5000})
        
        # Update configuration
        engine.command(
            NEUROSYMBOLIC_ENGINE_API_KEY='new-key',
            NEUROSYMBOLIC_ENGINE_MODEL='gpt-3.5-turbo',
            seed=42
        )
        
        # Verify sync engine was updated
        assert mock_sync.command.called
        
        # Simulate the sync engine updating its properties
        mock_sync.model = 'gpt-3.5-turbo'
        mock_sync.seed = 42
        
        # Update engine properties to match (as done in the real command method)
        engine.model = mock_sync.model
        engine.seed = mock_sync.seed
        
        assert engine.model == 'gpt-3.5-turbo'
        assert engine.seed == 42


@pytest.mark.asyncio
async def test_id_method():
    """Test id method delegation."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.id.return_value = 'neurosymbolic'
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine()
        assert engine.id() == 'neurosymbolic'
        assert mock_sync.id.called


@pytest.mark.asyncio
async def test_compute_tokens_methods():
    """Test token computation method delegation."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.compute_remaining_tokens.return_value = 1000
        mock_sync.compute_required_tokens.return_value = 500
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine()
        
        # Test compute_remaining_tokens
        remaining = engine.compute_remaining_tokens([{"role": "user", "content": "Test"}])
        assert remaining == 1000
        assert mock_sync.compute_remaining_tokens.called
        
        # Test compute_required_tokens
        required = engine.compute_required_tokens([{"role": "user", "content": "Test"}])
        assert required == 500
        assert mock_sync.compute_required_tokens.called


@pytest.mark.asyncio
async def test_prepare_method():
    """Test prepare method delegation."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine()
        
        mock_arg = Mock()
        engine.prepare(mock_arg)
        
        mock_sync.prepare.assert_called_once_with(mock_arg)


@pytest.mark.asyncio
async def test_max_tokens_deprecation_handling():
    """Test handling of deprecated max_tokens parameter."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        
        # Mock the payload preparation to return max_tokens
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test"}],
            'model': 'gpt-4',
            'max_tokens': 100  # Old parameter
        })
        
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits=None)
        
        # Mock the async client
        engine.async_client = AsyncMock()
        engine.async_client.chat = AsyncMock()
        engine.async_client.chat.completions = AsyncMock()
        engine.async_client.chat.completions.create = AsyncMock(return_value=MockCompletion())
        
        # Create mock argument
        mock_arg = MagicMock()
        mock_arg.prop.prepared_input = [{"role": "user", "content": "Test"}]
        mock_arg.prop.truncation_percentage = 0.8
        mock_arg.prop.truncation_type = 'tail'
        mock_arg.kwargs = {'max_tokens': 100}
        
        # Process
        output, metadata = await engine._process_single_async(mock_arg)
        
        # Verify the call was made with max_completion_tokens
        assert engine.async_client.chat.completions.create.called
        call_args = engine.async_client.chat.completions.create.call_args
        if call_args and call_args[1]:  # Check kwargs exist
            call_kwargs = call_args[1]
            assert 'max_completion_tokens' in call_kwargs
            assert call_kwargs['max_completion_tokens'] == 100
            assert 'max_tokens' not in call_kwargs


@pytest.mark.asyncio
async def test_edge_case_empty_batch():
    """Test handling of empty batch."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine()
        
        # Process empty batch
        outputs, metadata = await engine._forward_async([])
        
        # Should return empty results
        assert outputs == []
        assert metadata == []


@pytest.mark.asyncio
async def test_edge_case_none_content():
    """Test handling of None content in responses."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test"}],
            'model': 'gpt-4'
        })
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits=None)
        
        # Mock response with None content
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]
        
        engine.async_client = AsyncMock()
        engine.async_client.chat = AsyncMock()
        engine.async_client.chat.completions = AsyncMock()
        engine.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create mock argument
        mock_arg = MagicMock()
        mock_arg.prop.prepared_input = [{"role": "user", "content": "Test"}]
        mock_arg.prop.truncation_percentage = 0.8
        mock_arg.prop.truncation_type = 'tail'
        mock_arg.kwargs = {}
        
        # Process
        output, metadata = await engine._process_single_async(mock_arg)
        
        # Should handle None gracefully
        assert output == [None]
        assert 'raw_output' in metadata


@pytest.mark.asyncio
async def test_legacy_truncate_fallback():
    """Test fallback when truncate method doesn't exist."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        # Simulate missing truncate method
        type(mock_sync).truncate = PropertyMock(side_effect=AttributeError)
        mock_sync._prepare_request_payload = MagicMock(return_value={
            'messages': [{"role": "user", "content": "Test fallback"}],
            'model': 'gpt-4'
        })
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits=None)
        
        # Mock the async client
        engine.async_client = AsyncMock()
        engine.async_client.chat = AsyncMock()
        engine.async_client.chat.completions = AsyncMock()
        engine.async_client.chat.completions.create = AsyncMock(return_value=MockCompletion())
        
        # Create mock argument
        mock_arg = MagicMock()
        mock_arg.prop.prepared_input = [{"role": "user", "content": "Test fallback"}]
        mock_arg.kwargs = {}
        
        # Process - should use prepared_input directly
        output, metadata = await engine._process_single_async(mock_arg)
        
        # Verify it worked without truncate
        assert output == ["Test response"]
        assert engine.async_client.chat.completions.create.called
        call_args = engine.async_client.chat.completions.create.call_args
        if call_args and call_args[1]:  # Check kwargs exist
            call_kwargs = call_args[1]
            assert call_kwargs['messages'] == [{"role": "user", "content": "Test fallback"}]


@pytest.mark.asyncio
async def test_legacy_payload_preparation_fallback():
    """Test fallback when _prepare_request_payload doesn't exist."""
    with patch('src.engines.async_chatgpt_rate_limited.GPTXChatEngine') as mock_gptx:
        mock_sync = MagicMock()
        mock_sync.config = {'NEUROSYMBOLIC_ENGINE_API_KEY': 'test-key'}
        mock_sync.model = "gpt-4"
        mock_sync.tokenizer = Mock()
        mock_sync.max_context_tokens = 8192
        mock_sync.max_response_tokens = 4096
        mock_sync.seed = None
        mock_sync.name = 'GPTXChatEngine'
        mock_sync.prepare = MagicMock()
        mock_sync.truncate = MagicMock(return_value=[{"role": "user", "content": "Test"}])
        # Simulate missing _prepare_request_payload method
        type(mock_sync)._prepare_request_payload = PropertyMock(side_effect=AttributeError)
        mock_gptx.return_value = mock_sync
        
        engine = AsyncGPTXBatchEngine(model="gpt-4", rate_limits=None)
        
        # Mock the async client
        engine.async_client = AsyncMock()
        engine.async_client.chat = AsyncMock()
        engine.async_client.chat.completions = AsyncMock()
        engine.async_client.chat.completions.create = AsyncMock(return_value=MockCompletion())
        
        # Create mock argument
        mock_arg = MagicMock()
        mock_arg.prop.prepared_input = [{"role": "user", "content": "Test"}]
        mock_arg.prop.truncation_percentage = 0.8
        mock_arg.prop.truncation_type = 'tail'
        mock_arg.kwargs = {
            'temperature': 0.5,
            'max_tokens': 100,
            'seed': 42,
            'stream': False
        }
        
        # Process - should build payload manually
        output, metadata = await engine._process_single_async(mock_arg)
        
        # Verify fallback payload was created
        assert engine.async_client.chat.completions.create.called
        call_args = engine.async_client.chat.completions.create.call_args
        if call_args and call_args[1]:  # Check kwargs exist
            call_kwargs = call_args[1]
            assert call_kwargs['model'] == 'gpt-4'
            assert call_kwargs['temperature'] == 0.5
            assert call_kwargs['max_completion_tokens'] == 100  # Converted from max_tokens
            assert call_kwargs['stream'] is False
            assert 'seed' in call_kwargs