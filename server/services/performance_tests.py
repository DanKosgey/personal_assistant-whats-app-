"""
Performance tests for EOC detection and conversation summarization
"""

import os
import time
import logging
import asyncio
import random
import string
from typing import List, Dict, Any
from datetime import datetime
import json

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .eoc_detector import detect_eoc_with_embedding, embed_text
from .eoc_classifier import classify_eoc
from .advanced_memory import AdvancedMemoryManager

logger = logging.getLogger(__name__)

class PerformanceTester:
    """Performance testing suite for EOC system components"""
    
    def __init__(self):
        self.memory_manager = AdvancedMemoryManager()
    
    def generate_test_transcript(self, length_chars: int = 1000) -> str:
        """
        Generate a test conversation transcript of specified length.
        
        Args:
            length_chars: Desired length in characters
            
        Returns:
            Generated transcript
        """
        # Sample conversation parts
        conversation_parts = [
            "Hello, how can I help you today?",
            "I need assistance with my account.",
            "Sure, I can help with that. What seems to be the issue?",
            "I'm having trouble accessing my profile settings.",
            "Let me check that for you. Can you confirm your account number?",
            "Yes, it's ACCT-12345-XYZ.",
            "Thank you. I'm looking up your account now.",
            "I appreciate your help with this.",
            "No problem at all. I found your account.",
            "Great! What information can you provide?",
            "I can see that your settings are currently set to default.",
            "That explains the issues I've been experiencing.",
            "Would you like me to update your preferences?",
            "Yes, please update them to match my requirements.",
            "I've updated your settings. Is there anything else I can help with?",
            "No, that's all for now. Thank you for your assistance.",
            "You're welcome. Have a great day!",
            "You too. Goodbye!"
        ]
        
        # Build transcript until desired length
        transcript = ""
        while len(transcript) < length_chars:
            # Add random parts
            part = random.choice(conversation_parts)
            transcript += f"User: {part}\nAssistant: "
            
            # Add response
            response = random.choice(conversation_parts)
            transcript += f"{response}\n\n"
        
        # Trim to exact length if needed
        return transcript[:length_chars]
    
    async def measure_eoc_latency(self, num_tests: int = 100) -> Dict[str, Any]:
        """
        Measure EOC detection latency.
        
        Args:
            num_tests: Number of test iterations
            
        Returns:
            Dictionary with latency statistics
        """
        logger.info(f"Starting EOC latency test with {num_tests} iterations")
        
        # Test data
        test_messages = [
            "Thank you for your help",
            "Thanks, that's all I needed",
            "Goodbye, see you later",
            "I appreciate your assistance",
            "That's everything, bye",
            "I need help with something else",  # Non-EOC example
            "Can you provide more information?",  # Non-EOC example
            "What are your hours of operation?",  # Non-EOC example
        ]
        
        # Add some random variations
        for i in range(num_tests - len(test_messages)):
            test_messages.append(random.choice(test_messages))
        
        # Metrics
        embedding_times = []
        classifier_times = []
        total_times = []
        
        try:
            # Test embedding-based detection
            logger.info("Testing embedding-based EOC detection...")
            for message in test_messages:
                start_time = time.perf_counter()
                
                try:
                    # Test embedding detection
                    embed_start = time.perf_counter()
                    is_eoc_embed, confidence_embed, example_id = detect_eoc_with_embedding(message)
                    embed_time = time.perf_counter() - embed_start
                    embedding_times.append(embed_time)
                    
                    # Test classifier
                    classifier_start = time.perf_counter()
                    is_eoc_class, probability = classify_eoc(message, confidence_embed, confidence_embed)
                    classifier_time = time.perf_counter() - classifier_start
                    classifier_times.append(classifier_time)
                    
                except Exception as e:
                    logger.warning(f"Error in EOC detection for message '{message[:30]}...': {e}")
                    continue
                finally:
                    total_time = time.perf_counter() - start_time
                    total_times.append(total_time)
            
            # Calculate statistics
            if total_times:
                avg_total_time = sum(total_times) / len(total_times)
                min_total_time = min(total_times)
                max_total_time = max(total_times)
                
                avg_embedding_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
                avg_classifier_time = sum(classifier_times) / len(classifier_times) if classifier_times else 0
                
                # Calculate percentiles
                sorted_times = sorted(total_times)
                p50 = sorted_times[len(sorted_times) // 2]
                p95 = sorted_times[int(len(sorted_times) * 0.95)]
                p99 = sorted_times[int(len(sorted_times) * 0.99)]
                
                results = {
                    'test_type': 'eoc_detection_latency',
                    'test_date': datetime.utcnow().isoformat(),
                    'num_tests': len(total_times),
                    'average_latency_seconds': avg_total_time,
                    'min_latency_seconds': min_total_time,
                    'max_latency_seconds': max_total_time,
                    'median_latency_seconds': p50,
                    'p95_latency_seconds': p95,
                    'p99_latency_seconds': p99,
                    'average_embedding_time': avg_embedding_time,
                    'average_classifier_time': avg_classifier_time,
                    'latency_under_100ms': len([t for t in total_times if t < 0.1]) / len(total_times) if total_times else 0,
                    'latency_under_50ms': len([t for t in total_times if t < 0.05]) / len(total_times) if total_times else 0
                }
                
                logger.info(f"EOC detection test completed. Average latency: {avg_total_time:.4f}s")
                return results
            else:
                logger.error("No successful EOC detection tests completed")
                return {
                    'test_type': 'eoc_detection_latency',
                    'test_date': datetime.utcnow().isoformat(),
                    'error': 'No successful tests completed'
                }
                
        except Exception as e:
            logger.error(f"Error in EOC latency test: {e}")
            return {
                'test_type': 'eoc_detection_latency',
                'test_date': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    async def measure_summarization_time(self, num_tests: int = 10) -> Dict[str, Any]:
        """
        Measure conversation summarization time.
        
        Args:
            num_tests: Number of test iterations
            
        Returns:
            Dictionary with summarization time statistics
        """
        logger.info(f"Starting summarization time test with {num_tests} iterations")
        
        # Test with different transcript lengths
        test_lengths = [500, 1000, 2000, 5000, 10000]
        results_by_length = {}
        
        try:
            for length in test_lengths:
                logger.info(f"Testing summarization for {length} character transcripts")
                
                summarization_times = []
                hierarchical_times = []
                
                for i in range(num_tests):
                    try:
                        # Generate test transcript
                        transcript = self.generate_test_transcript(length)
                        user_id = f"test_user_{i}"
                        
                        # Test regular summarization
                        start_time = time.perf_counter()
                        summary = await self.memory_manager.summarize_conversation(user_id)
                        summary_time = time.perf_counter() - start_time
                        summarization_times.append(summary_time)
                        
                        # Test hierarchical summarization for longer transcripts
                        if length > 1500:
                            start_time = time.perf_counter()
                            hierarchical_summary = await self.memory_manager.hierarchical_summarize_conversation(user_id)
                            hierarchical_time = time.perf_counter() - start_time
                            hierarchical_times.append(hierarchical_time)
                            
                    except Exception as e:
                        logger.warning(f"Error in summarization test for length {length}: {e}")
                        continue
                
                # Calculate statistics for this length
                if summarization_times:
                    avg_summary_time = sum(summarization_times) / len(summarization_times)
                    min_summary_time = min(summarization_times)
                    max_summary_time = max(summarization_times)
                    
                    # Percentiles
                    sorted_times = sorted(summarization_times)
                    p50 = sorted_times[len(sorted_times) // 2]
                    p95 = sorted_times[int(len(sorted_times) * 0.95)]
                    
                    length_results = {
                        'transcript_length': length,
                        'num_tests': len(summarization_times),
                        'average_time_seconds': avg_summary_time,
                        'min_time_seconds': min_summary_time,
                        'max_time_seconds': max_summary_time,
                        'median_time_seconds': p50,
                        'p95_time_seconds': p95,
                        'time_under_1s': len([t for t in summarization_times if t < 1.0]) / len(summarization_times)
                    }
                    
                    # Add hierarchical results if available
                    if hierarchical_times:
                        avg_hierarchical_time = sum(hierarchical_times) / len(hierarchical_times)
                        length_results['hierarchical_average_time_seconds'] = avg_hierarchical_time
                        length_results['hierarchical_tests'] = len(hierarchical_times)
                    
                    results_by_length[str(length)] = length_results
                    
                    logger.info(f"Summarization test for {length} chars: avg {avg_summary_time:.4f}s")
                else:
                    results_by_length[str(length)] = {
                        'transcript_length': length,
                        'error': 'No successful tests completed'
                    }
            
            # Overall results
            overall_results = {
                'test_type': 'summarization_time',
                'test_date': datetime.utcnow().isoformat(),
                'results_by_length': results_by_length
            }
            
            return overall_results
            
        except Exception as e:
            logger.error(f"Error in summarization time test: {e}")
            return {
                'test_type': 'summarization_time',
                'test_date': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    async def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """
        Run comprehensive performance tests.
        
        Returns:
            Dictionary with all test results
        """
        logger.info("Starting comprehensive performance test suite")
        
        results = {
            'test_suite': 'eoc_performance_tests',
            'start_time': datetime.utcnow().isoformat()
        }
        
        try:
            # Run EOC detection latency test
            eoc_results = await self.measure_eoc_latency(num_tests=100)
            results['eoc_detection'] = eoc_results
            
            # Run summarization time test
            summary_results = await self.measure_summarization_time(num_tests=5)
            results['summarization'] = summary_results
            
            # Add overall summary
            results['end_time'] = datetime.utcnow().isoformat()
            results['duration_seconds'] = (
                datetime.fromisoformat(results['end_time'].replace('Z', '+00:00')) - 
                datetime.fromisoformat(results['start_time'].replace('Z', '+00:00'))
            ).total_seconds()
            
            logger.info("Comprehensive performance test suite completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive performance test: {e}")
            results['error'] = str(e)
            results['end_time'] = datetime.utcnow().isoformat()
            return results
    
    async def export_performance_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        """
        Export performance test results to a JSON file.
        
        Args:
            results: Performance test results dictionary
            output_file: Optional output file path
            
        Returns:
            Path to the exported file
        """
        try:
            # Determine output file path
            if output_file is None:
                output_file = os.path.join(
                    os.getcwd(), 
                    f"performance_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                )
            
            # Export to file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Performance test results exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting performance results: {e}")
            raise

# Global performance tester instance
perf_tester = PerformanceTester()

async def measure_eoc_latency(num_tests: int = 100) -> Dict[str, Any]:
    """
    Measure EOC detection latency.
    
    Args:
        num_tests: Number of test iterations
        
    Returns:
        Dictionary with latency statistics
    """
    return await perf_tester.measure_eoc_latency(num_tests)

async def measure_summarization_time(num_tests: int = 10) -> Dict[str, Any]:
    """
    Measure conversation summarization time.
    
    Args:
        num_tests: Number of test iterations
        
    Returns:
        Dictionary with summarization time statistics
    """
    return await perf_tester.measure_summarization_time(num_tests)

async def run_comprehensive_performance_test() -> Dict[str, Any]:
    """
    Run comprehensive performance tests.
    
    Returns:
        Dictionary with all test results
    """
    return await perf_tester.run_comprehensive_performance_test()

async def export_performance_results(results: Dict[str, Any], output_file: str = None) -> str:
    """
    Export performance test results to a file.
    
    Args:
        results: Performance test results dictionary
        output_file: Optional output file path
        
    Returns:
        Path to the exported file
    """
    return await perf_tester.export_performance_results(results, output_file)

async def run_performance_test_suite():
    """
    Run the complete performance test suite and generate report.
    """
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting performance test suite...")
        
        # Run comprehensive tests
        results = await run_comprehensive_performance_test()
        
        # Export results
        export_path = await export_performance_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE TEST RESULTS SUMMARY")
        print("="*60)
        
        if 'eoc_detection' in results:
            eoc_results = results['eoc_detection']
            print(f"\nEOC Detection Performance:")
            print(f"  Average Latency: {eoc_results.get('average_latency_seconds', 0):.4f}s")
            print(f"  Median Latency: {eoc_results.get('median_latency_seconds', 0):.4f}s")
            print(f"  95th Percentile: {eoc_results.get('p95_latency_seconds', 0):.4f}s")
            print(f"  Under 100ms: {eoc_results.get('latency_under_100ms', 0)*100:.1f}%")
            print(f"  Tests Completed: {eoc_results.get('num_tests', 0)}")
        
        if 'summarization' in results:
            summary_results = results['summarization']
            print(f"\nSummarization Performance:")
            for length, data in summary_results.get('results_by_length', {}).items():
                if 'error' not in data:
                    print(f"  {length} chars:")
                    print(f"    Average Time: {data.get('average_time_seconds', 0):.4f}s")
                    print(f"    Median Time: {data.get('median_time_seconds', 0):.4f}s")
                    print(f"    Under 1s: {data.get('time_under_1s', 0)*100:.1f}%")
        
        print(f"\nResults exported to: {export_path}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running performance test suite: {e}")
        raise

if __name__ == "__main__":
    # Run the performance test suite
    asyncio.run(run_performance_test_suite())