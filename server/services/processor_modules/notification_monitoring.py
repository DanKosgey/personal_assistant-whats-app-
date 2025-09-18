"""
Monitoring and alerting for the autonomous notification system.
This module provides metrics collection and alerting capabilities.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# In-memory metrics storage (in production, you would use a proper metrics system like Prometheus)
_metrics = {
    "notifications_sent_total": 0,
    "notifications_failed_total": 0,
    "notifications_suppressed_total": 0,
    "notification_feedback_helpful": 0,
    "notification_feedback_not_helpful": 0,
    "avg_importance_score_for_sent": [],
    "notifications_by_owner": {},
    "notification_latencies": []
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "false_positive_rate_threshold": 0.20,  # 20% false positive rate
    "alert_time_window": 24 * 3600,  # 24 hours in seconds
    "high_failure_rate_threshold": 0.10,  # 10% failure rate
    "low_feedback_rate_threshold": 0.05  # 5% feedback rate
}

def record_notification_sent(owner_id: str, importance_score: float, latency: float = 0.0):
    """
    Record that a notification was sent.
    """
    try:
        _metrics["notifications_sent_total"] += 1
        _metrics["avg_importance_score_for_sent"].append(importance_score)
        
        # Track notifications by owner
        if owner_id not in _metrics["notifications_by_owner"]:
            _metrics["notifications_by_owner"][owner_id] = 0
        _metrics["notifications_by_owner"][owner_id] += 1
        
        # Track latency
        if latency > 0:
            _metrics["notification_latencies"].append(latency)
            
        logger.info(f"Notification sent to owner {owner_id} with importance score {importance_score}")
    except Exception as e:
        logger.error(f"Failed to record notification sent: {e}")

def record_notification_failed(reason: str = "unknown"):
    """
    Record that a notification failed to send.
    """
    try:
        _metrics["notifications_failed_total"] += 1
        logger.warning(f"Notification failed to send: {reason}")
    except Exception as e:
        logger.error(f"Failed to record notification failure: {e}")

def record_notification_suppressed(reason: str = "unknown"):
    """
    Record that a notification was suppressed.
    """
    try:
        _metrics["notifications_suppressed_total"] += 1
        logger.info(f"Notification suppressed: {reason}")
    except Exception as e:
        logger.error(f"Failed to record notification suppression: {e}")

def record_notification_feedback(notification_id: str, owner_id: str, helpful: bool, notes: str = ""):
    """
    Record owner feedback on a notification.
    """
    try:
        if helpful:
            _metrics["notification_feedback_helpful"] += 1
        else:
            _metrics["notification_feedback_not_helpful"] += 1
            
        logger.info(f"Notification feedback recorded for {notification_id}: {'helpful' if helpful else 'not helpful'}")
    except Exception as e:
        logger.error(f"Failed to record notification feedback: {e}")

def get_notification_metrics() -> Dict[str, Any]:
    """
    Get current notification metrics.
    """
    try:
        # Calculate average importance score
        avg_importance_score = 0.0
        if _metrics["avg_importance_score_for_sent"]:
            avg_importance_score = sum(_metrics["avg_importance_score_for_sent"]) / len(_metrics["avg_importance_score_for_sent"])
        
        # Calculate false positive rate
        false_positive_rate = 0.0
        total_feedback = _metrics["notification_feedback_helpful"] + _metrics["notification_feedback_not_helpful"]
        if total_feedback > 0:
            false_positive_rate = _metrics["notification_feedback_not_helpful"] / total_feedback
        
        # Calculate average latency
        avg_latency = 0.0
        if _metrics["notification_latencies"]:
            avg_latency = sum(_metrics["notification_latencies"]) / len(_metrics["notification_latencies"])
        
        return {
            "notifications_sent_total": _metrics["notifications_sent_total"],
            "notifications_failed_total": _metrics["notifications_failed_total"],
            "notifications_suppressed_total": _metrics["notifications_suppressed_total"],
            "notification_feedback_helpful": _metrics["notification_feedback_helpful"],
            "notification_feedback_not_helpful": _metrics["notification_feedback_not_helpful"],
            "total_feedback": total_feedback,
            "avg_importance_score_for_sent": round(avg_importance_score, 3),
            "false_positive_rate": round(false_positive_rate, 3),
            "avg_notification_latency": round(avg_latency, 3),
            "notifications_by_owner": _metrics["notifications_by_owner"]
        }
    except Exception as e:
        logger.error(f"Failed to get notification metrics: {e}")
        return {}

def check_alerts() -> Dict[str, Any]:
    """
    Check if any alerts should be triggered based on current metrics.
    """
    try:
        metrics = get_notification_metrics()
        alerts = []
        
        # Check false positive rate
        if metrics.get("false_positive_rate", 0) > ALERT_THRESHOLDS["false_positive_rate_threshold"]:
            alerts.append({
                "type": "high_false_positive_rate",
                "message": f"False positive rate is {metrics['false_positive_rate']:.2%}, exceeding threshold of {ALERT_THRESHOLDS['false_positive_rate_threshold']:.2%}",
                "severity": "warning"
            })
        
        # Check failure rate
        total_notifications = metrics.get("notifications_sent_total", 0) + metrics.get("notifications_failed_total", 0)
        if total_notifications > 0:
            failure_rate = metrics.get("notifications_failed_total", 0) / total_notifications
            if failure_rate > ALERT_THRESHOLDS["high_failure_rate_threshold"]:
                alerts.append({
                    "type": "high_failure_rate",
                    "message": f"Notification failure rate is {failure_rate:.2%}, exceeding threshold of {ALERT_THRESHOLDS['high_failure_rate_threshold']:.2%}",
                    "severity": "warning"
                })
        
        # Check feedback rate
        total_sent = metrics.get("notifications_sent_total", 0)
        total_feedback = metrics.get("total_feedback", 0)
        if total_sent > 0:
            feedback_rate = total_feedback / total_sent
            if feedback_rate < ALERT_THRESHOLDS["low_feedback_rate_threshold"]:
                alerts.append({
                    "type": "low_feedback_rate",
                    "message": f"Notification feedback rate is {feedback_rate:.2%}, below threshold of {ALERT_THRESHOLDS['low_feedback_rate_threshold']:.2%}",
                    "severity": "info"
                })
        
        return {
            "alerts": alerts,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to check alerts: {e}")
        return {
            "alerts": [],
            "metrics": {}
        }

def reset_metrics():
    """
    Reset all metrics (useful for testing).
    """
    global _metrics
    _metrics = {
        "notifications_sent_total": 0,
        "notifications_failed_total": 0,
        "notifications_suppressed_total": 0,
        "notification_feedback_helpful": 0,
        "notification_feedback_not_helpful": 0,
        "avg_importance_score_for_sent": [],
        "notifications_by_owner": {},
        "notification_latencies": []
    }

# Example usage
if __name__ == "__main__":
    # Example of how to use the monitoring functions
    print("Notification Monitoring System")
    print("----------------------------")
    
    # Record some metrics
    record_notification_sent("owner_123", 0.85, 0.12)
    record_notification_sent("owner_123", 0.72, 0.08)
    record_notification_failed("provider_error")
    record_notification_suppressed("duplicate")
    record_notification_feedback("notif_abc", "owner_123", True, "Very useful summary")
    record_notification_feedback("notif_def", "owner_123", False, "Not relevant to me")
    
    # Get metrics
    metrics = get_notification_metrics()
    print("Current Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Check alerts
    alerts = check_alerts()
    if alerts["alerts"]:
        print("\nAlerts:")
        for alert in alerts["alerts"]:
            print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
    else:
        print("\nNo alerts triggered")