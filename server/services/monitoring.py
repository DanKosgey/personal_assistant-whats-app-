"""
Monitoring and Metrics Collection for WhatsApp AI Agent
Implements simple metric counters for EOC detection and other key events
"""

import logging
from typing import Dict, Any
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Simple metrics collector for EOC and other events"""
    
    def __init__(self, metrics_file: str = "metrics.log"):
        self.metrics_file = metrics_file
        self.counters = {
            "eoc_detected": 0,
            "eoc_confirmed": 0,
            "summary_sent": 0,
            "feedback_received": 0,
            "conversations_ended": 0,
            "conversations_reopened": 0,
            "messages_processed": 0,
            "ai_calls": 0,
            "errors": 0
        }
        self.timings = {
            "avg_eoc_detection_time": 0.0,
            "avg_summary_generation_time": 0.0,
            "avg_response_time": 0.0
        }
        logger.info(f"Metrics collector initialized with file: {metrics_file}")
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter by value"""
        if counter_name in self.counters:
            self.counters[counter_name] += value
            self._log_metrics()
        else:
            logger.warning(f"Unknown counter: {counter_name}")
    
    def update_timing(self, timing_name: str, value: float):
        """Update a timing metric (moving average)"""
        if timing_name in self.timings:
            # Simple moving average
            current_avg = self.timings[timing_name]
            # For simplicity, we'll just update with the new value
            # In a production system, you might want a proper moving average
            self.timings[timing_name] = value
            self._log_metrics()
        else:
            logger.warning(f"Unknown timing metric: {timing_name}")
    
    def record_eoc_detection(self, confidence: float, method: str):
        """Record an EOC detection event"""
        self.increment_counter("eoc_detected")
        logger.info(f"EOC detected - Confidence: {confidence:.3f}, Method: {method}")
    
    def record_eoc_confirmation(self):
        """Record an EOC confirmation event"""
        self.increment_counter("eoc_confirmed")
        logger.info("EOC confirmed")
    
    def record_summary_sent(self):
        """Record a summary sent event"""
        self.increment_counter("summary_sent")
        logger.info("Summary sent to owner")
    
    def record_feedback(self, feedback_type: str):
        """Record a feedback event"""
        self.increment_counter("feedback_received")
        logger.info(f"Feedback received: {feedback_type}")
    
    def record_conversation_ended(self):
        """Record a conversation ended event"""
        self.increment_counter("conversations_ended")
        logger.info("Conversation ended")
    
    def record_conversation_reopened(self):
        """Record a conversation reopened event"""
        self.increment_counter("conversations_reopened")
        logger.info("Conversation reopened")
    
    def record_message_processed(self):
        """Record a message processed event"""
        self.increment_counter("messages_processed")
        logger.debug("Message processed")
    
    def record_ai_call(self):
        """Record an AI call event"""
        self.increment_counter("ai_calls")
        logger.debug("AI call made")
    
    def record_error(self, error_type: str):
        """Record an error event"""
        self.increment_counter("errors")
        logger.error(f"Error recorded: {error_type}")
    
    def _log_metrics(self):
        """Log current metrics to file"""
        try:
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "counters": self.counters.copy(),
                "timings": self.timings.copy()
            }
            
            # Append to metrics file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics_data) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "counters": self.counters.copy(),
            "timings": self.timings.copy(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def reset_counters(self):
        """Reset all counters to zero"""
        for key in self.counters:
            self.counters[key] = 0
        logger.info("Metrics counters reset")

# Global instance
metrics_collector = MetricsCollector()

def get_metrics() -> Dict[str, Any]:
    """Get current metrics"""
    return metrics_collector.get_metrics()

def record_eoc_detection(confidence: float, method: str):
    """Record an EOC detection event"""
    metrics_collector.record_eoc_detection(confidence, method)

def record_eoc_confirmation():
    """Record an EOC confirmation event"""
    metrics_collector.record_eoc_confirmation()

def record_summary_sent():
    """Record a summary sent event"""
    metrics_collector.record_summary_sent()

def record_feedback(feedback_type: str):
    """Record a feedback event"""
    metrics_collector.record_feedback(feedback_type)

def record_conversation_ended():
    """Record a conversation ended event"""
    metrics_collector.record_conversation_ended()

def record_conversation_reopened():
    """Record a conversation reopened event"""
    metrics_collector.record_conversation_reopened()

def record_message_processed():
    """Record a message processed event"""
    metrics_collector.record_message_processed()

def record_ai_call():
    """Record an AI call event"""
    metrics_collector.record_ai_call()

def record_error(error_type: str):
    """Record an error event"""
    metrics_collector.record_error(error_type)