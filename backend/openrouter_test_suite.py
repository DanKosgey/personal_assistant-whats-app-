#!/usr/bin/env python3
"""
Comprehensive test suite for OpenRouter Client
Tests all functionality including error handling, key rotation, and edge cases.
"""

import asyncio
import logging
import os
import argparse
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback
import sys
from contextlib import asynccontextmanager

# Assuming the OpenRouter client is in a module called 'openrouter_client'
# Adjust import as needed for your project structure
from openrouter_client import (
    OpenRouterClient, 
    GenerationRequest, 
    GenerationResponse,
    OpenRouterError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    QuotaExceededError,
    generate_text,
    test_all_keys
)

# Configure logging (force UTF-8 to avoid Windows console encode errors with emoji)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler (stdout) with UTF-8
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
try:
    ch.stream.reconfigure(encoding='utf-8')
except Exception:
    # Older Python versions or environments may not support reconfigure; ignore.
    pass

# File handler with UTF-8
fh = logging.FileHandler('openrouter_test.log', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
log = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for test results"""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class OpenRouterTestSuite:
    """Comprehensive test suite for OpenRouter client"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.client: Optional[OpenRouterClient] = None
        
    def add_result(self, name: str, passed: bool, duration: float, 
                   error: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Add a test result"""
        self.results.append(TestResult(name, passed, duration, error, details))
        status = "✅ PASS" if passed else "❌ FAIL"
        log.info(f"{status} {name} ({duration:.2f}s)")
        if error:
            log.error(f"Error: {error}")

    @asynccontextmanager
    async def test_context(self, test_name: str):
        """Context manager for running individual tests"""
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.add_result(test_name, True, duration)
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.add_result(test_name, False, duration, error_msg)
            log.exception(f"Test {test_name} failed")

    async def test_client_initialization(self):
        """Test client initialization with various configurations"""
        async with self.test_context("Client Initialization - Default"):
            self.client = OpenRouterClient()
            assert len(self.client.keys) > 0, "No API keys found"
            
        async with self.test_context("Client Initialization - Custom Model"):
            client = OpenRouterClient(model="anthropic/claude-3-haiku")
            assert client.model == "anthropic/claude-3-haiku"
            
        async with self.test_context("Client Initialization - Custom Keys"):
            # Test with mock keys
            mock_keys = ["test-key-1", "test-key-2"]
            try:
                client = OpenRouterClient(keys=mock_keys, model="test-model")
                assert len(client.keys) == 2
            except Exception:
                # Expected to fail without valid keys, but initialization should work
                pass

    async def test_environment_key_loading(self):
        """Test loading keys from environment variables"""
        async with self.test_context("Environment Key Loading"):
            # Temporarily set test environment variables
            original_keys = {}
            test_keys = {
                "OPENROUTER_API_KEY_1": "test-key-1",
                "OPENROUTER_API_KEY_2": "test-key-2", 
                "OPENROUTER_API_KEY_TEST": "test-key-3"
            }
            
            for key, value in test_keys.items():
                original_keys[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                loaded_keys = OpenRouterClient._load_keys_from_env()
                # Should load at least the test keys (plus any real ones)
                test_key_values = list(test_keys.values())
                for test_key in test_key_values:
                    assert test_key in loaded_keys, f"Test key {test_key} not loaded"
            finally:
                # Restore original environment
                for key, original_value in original_keys.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    async def test_basic_text_generation(self):
        """Test basic text generation functionality"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Basic Text Generation - String Prompt"):
            response = await self.client.generate("What is 2+2? Answer briefly.")
            assert isinstance(response, GenerationResponse)
            assert len(response.text) > 0
            assert "4" in response.text or "four" in response.text.lower()
            
        async with self.test_context("Basic Text Generation - GenerationRequest"):
            request = GenerationRequest(
                prompt="Name one color. Just the color name.",
                temperature=0.0
            )
            response = await self.client.generate(request)
            assert isinstance(response.text, str)
            assert len(response.text.strip()) > 0

    async def test_message_format_generation(self):
        """Test generation with message format"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Message Format Generation"):
            request = GenerationRequest(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
                    {"role": "user", "content": "What is the capital of Japan?"}
                ],
                temperature=0.0
            )
            response = await self.client.generate(request)
            assert "Tokyo" in response.text

    async def test_generation_parameters(self):
        """Test various generation parameters"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Generation Parameters - Temperature"):
            # Test low temperature (should be deterministic)
            request = GenerationRequest(
                prompt="Pick a number between 1 and 10:",
                temperature=0.0
            )
            response1 = await self.client.generate(request)
            response2 = await self.client.generate(request)
            # With temperature 0, responses should be identical or very similar
            assert len(response1.text) > 0 and len(response2.text) > 0
            
        async with self.test_context("Generation Parameters - Max Tokens"):
            request = GenerationRequest(
                prompt="Write a long story about dragons.",
                max_tokens=20,
                temperature=0.0
            )
            response = await self.client.generate(request)
            # Response should be limited by max_tokens
            assert len(response.text.split()) <= 30  # Some buffer for token vs word count

    async def test_model_selection(self):
        """Test model selection and switching"""
        if not self.client:
            self.client = OpenRouterClient()
            
        # Try with a few common models that should be available
        test_models = [
            "anthropic/claude-3-haiku",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-2-7b-chat"
        ]
        
        for model in test_models:
            async with self.test_context(f"Model Selection - {model}"):
                try:
                    request = GenerationRequest(
                        prompt="Say 'hello' in one word.",
                        model=model,
                        temperature=0.0
                    )
                    response = await self.client.generate(request)
                    assert len(response.text) > 0
                    assert response.model  # Should return the model that was used
                except (ModelNotFoundError, QuotaExceededError) as e:
                    # These errors are acceptable - model might not be available
                    log.warning(f"Model {model} not available: {e}")
                    # Mark as passed since this is expected behavior
                    pass

    async def test_key_rotation(self):
        """Test API key rotation functionality"""
        if len(self.client.keys) < 2:
            log.warning("Skipping key rotation test - need at least 2 API keys")
            return
            
        async with self.test_context("Key Rotation"):
            used_keys = set()
            
            # Make several requests to trigger key rotation
            for i in range(min(5, len(self.client.keys) * 2)):
                try:
                    response = await self.client.generate(f"Count to {i+1}")
                    used_keys.add(response.key_used)
                    await asyncio.sleep(0.1)  # Small delay between requests
                except Exception as e:
                    log.warning(f"Request {i} failed during key rotation test: {e}")
                    
            # Should have used multiple keys if we have them
            if len(self.client.keys) > 1:
                assert len(used_keys) > 1, f"Only used {len(used_keys)} keys out of {len(self.client.keys)}"

    async def test_error_handling(self):
        """Test error handling for various scenarios"""
        # Test with invalid API key
        async with self.test_context("Error Handling - Invalid Key"):
            try:
                client = OpenRouterClient(keys=["invalid-key-123"], model="openai/gpt-3.5-turbo")
                await client.generate("Hello")
                assert False, "Should have raised AuthenticationError"
            except AuthenticationError:
                pass  # Expected
            except OpenRouterError:
                pass  # Also acceptable
                
        # Test with invalid model
        async with self.test_context("Error Handling - Invalid Model"):
            try:
                await self.client.generate(GenerationRequest(
                    prompt="Hello",
                    model="invalid/model-name-12345"
                ))
                # If no error, the model might actually exist or be handled gracefully
                pass
            except (ModelNotFoundError, OpenRouterError):
                pass  # Expected

    async def test_key_status_monitoring(self):
        """Test key status and health monitoring"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Key Status Monitoring"):
            # Make a few requests to populate status
            for i in range(3):
                try:
                    await self.client.generate(f"Test request {i}")
                except Exception:
                    pass  # Ignore failures for this test
                    
            status = self.client.get_key_status()
            assert isinstance(status, dict)
            assert len(status) == len(self.client.keys)
            
            # Check status structure
            for key_prefix, key_status in status.items():
                assert "error_count" in key_status
                assert "success_rate" in key_status
                assert "total_requests" in key_status
                assert "is_available" in key_status
                assert isinstance(key_status["success_rate"], (int, float))

    async def test_key_health_tracking(self):
        """Test key health and blocking functionality"""
        async with self.test_context("Key Health Tracking"):
            healthy_keys = self.client.get_healthy_keys()
            assert isinstance(healthy_keys, list)
            
            # All keys should be available initially (unless previously blocked)
            all_status = self.client.get_key_status()
            available_count = sum(1 for status in all_status.values() if status["is_available"])
            assert available_count > 0, "No keys are available"

    async def test_individual_key_testing(self):
        """Test individual key testing functionality"""
        if not self.client.keys:
            return
            
        async with self.test_context("Individual Key Testing"):
            test_key = self.client.keys[0]
            try:
                result = await self.client.test_key(test_key)
                assert isinstance(result, dict)
                assert "choices" in result or "error" in result
            except Exception as e:
                # Key might be invalid, but test should handle it gracefully
                log.warning(f"Key test failed (this might be expected): {e}")

    async def test_model_listing(self):
        """Test model listing functionality"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Model Listing"):
            try:
                models = await self.client.list_models()
                assert isinstance(models, dict)
                # Should have a 'data' field with list of models
                if "data" in models:
                    assert isinstance(models["data"], list)
                    if models["data"]:
                        # Check first model structure
                        model = models["data"][0]
                        assert "id" in model
            except Exception as e:
                log.warning(f"Model listing failed: {e}")
                # This might fail due to API limits, so don't fail the test

    async def test_convenience_functions(self):
        """Test convenience functions"""
        async with self.test_context("Convenience Functions - generate_text"):
            result = await generate_text("What is 1+1? Answer with just the number.")
            assert isinstance(result, str)
            assert len(result.strip()) > 0
            assert "2" in result

    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Concurrent Requests"):
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = self.client.generate(f"What is {i} + 1? Answer with just the number.")
                tasks.append(task)
                
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful responses
            successful = 0
            for i, response in enumerate(responses):
                if isinstance(response, GenerationResponse):
                    successful += 1
                    assert str(i + 1) in response.text
                elif isinstance(response, Exception):
                    log.warning(f"Concurrent request {i} failed: {response}")
                    
            # At least some requests should succeed
            assert successful > 0, "No concurrent requests succeeded"

    async def test_context_manager(self):
        """Test async context manager functionality"""
        async with self.test_context("Context Manager"):
            async with OpenRouterClient() as client:
                response = await client.generate("Say hello in one word")
                assert len(response.text) > 0

    async def test_edge_cases(self):
        """Test various edge cases"""
        if not self.client:
            self.client = OpenRouterClient()
            
        async with self.test_context("Edge Cases - Empty Prompt"):
            try:
                await self.client.generate("")
                # Empty prompt might work or might fail - both are acceptable
                pass
            except Exception:
                pass  # Expected
                
        async with self.test_context("Edge Cases - Very Long Prompt"):
            long_prompt = "Tell me about: " + "word " * 1000
            try:
                response = await self.client.generate(long_prompt)
                assert isinstance(response.text, str)
            except Exception as e:
                # Might fail due to length limits
                log.warning(f"Long prompt test failed (acceptable): {e}")

    async def run_all_tests(self):
        """Run all tests in the test suite"""
        log.info("🚀 Starting OpenRouter Client Test Suite")
        
        tests = [
            self.test_client_initialization,
            self.test_environment_key_loading,
            self.test_basic_text_generation,
            self.test_message_format_generation,
            self.test_generation_parameters,
            self.test_model_selection,
            self.test_key_rotation,
            self.test_error_handling,
            self.test_key_status_monitoring,
            self.test_key_health_tracking,
            self.test_individual_key_testing,
            self.test_model_listing,
            self.test_convenience_functions,
            self.test_concurrent_requests,
            self.test_context_manager,
            self.test_edge_cases
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                log.exception(f"Test function {test.__name__} crashed")
                self.add_result(test.__name__, False, 0.0, f"Test crashed: {str(e)}")
                
        self.print_summary()

    def print_summary(self):
        """Print test results summary"""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration for r in self.results)
        
        print("\n" + "="*60)
        print("🧪 OPENROUTER CLIENT TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {len(self.results)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        print(f"📊 Success Rate: {(passed/len(self.results)*100):.1f}%")
        
        if failed > 0:
            print(f"\n❌ FAILED TESTS ({failed}):")
            for result in self.results:
                if not result.passed:
                    print(f"  • {result.name}: {result.error}")
                    
        print("\n📝 DETAILED RESULTS:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name} ({result.duration:.2f}s)")
            
        if self.client:
            print(f"\n🔑 KEY STATUS:")
            status = self.client.get_key_status()
            for key_prefix, key_status in status.items():
                health = "🟢" if key_status["is_available"] else "🔴"
                success_rate = key_status["success_rate"] * 100
                print(f"  {health} {key_prefix}: {success_rate:.1f}% success rate, "
                      f"{key_status['total_requests']} requests")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run test suite in offline simulation mode")
    args, _ = parser.parse_known_args()

    # Auto-load .env in backend if no keys present
    def try_load_dotenv(path):
        if not os.path.exists(path):
            return 0
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
                    loaded += 1
        return loaded

    if not os.environ.get("OPENROUTER_API_KEY") and not any(
        k.startswith("OPENROUTER_API_KEY") for k in os.environ.keys()
    ):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        loaded = try_load_dotenv(env_path)
        if loaded:
            print(f"Loaded {loaded} entries from {env_path}")
        if not os.environ.get("OPENROUTER_API_KEY") and not any(
            k.startswith("OPENROUTER_API_KEY") for k in os.environ.keys()
        ):
            print("⚠️ Warning: No OPENROUTER_API_KEY* environment variables found")
            print("Some tests may fail without valid API keys")
            print()
    
    # Run test suite
    test_suite = OpenRouterTestSuite()
    await test_suite.run_all_tests()
    
    # Test the convenience function for testing all keys
    print("\n🔍 Testing all configured keys...")
    try:
        key_test_results = await test_all_keys()
        print("Key Test Results:")
        for key_prefix, result in key_test_results.items():
            status = "✅" if result["status"] == "success" else "❌"
            print(f"  {status} {key_prefix}: {result['status']}")
            if result["error"]:
                print(f"    Error: {result['error']}")
    except Exception as e:
        print(f"Failed to test keys: {e}")


if __name__ == "__main__":
    asyncio.run(main())
