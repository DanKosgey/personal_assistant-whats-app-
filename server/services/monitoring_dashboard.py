"""
Comprehensive monitoring dashboard for EOC system
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .monitoring import get_metrics, METRICS_FILE
from ..database import db_manager

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Comprehensive monitoring dashboard for EOC system"""
    
    def __init__(self):
        self.refresh_interval = 60  # Refresh every 60 seconds
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics.
        
        Returns:
            Dictionary with system metrics
        """
        try:
            # Get current metrics
            current_metrics = get_metrics()
            
            # Get historical metrics
            historical_metrics = await self._get_historical_metrics()
            
            # Calculate trends
            trends = await self._calculate_trends(historical_metrics)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'current': current_metrics,
                'historical': historical_metrics,
                'trends': trends
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def _get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get historical metrics from the metrics file.
        
        Args:
            hours: Number of hours of historical data to retrieve
            
        Returns:
            List of historical metric entries
        """
        try:
            if not os.path.exists(METRICS_FILE):
                return []
            
            historical_data = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with open(METRICS_FILE, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                        if entry_time >= cutoff_time:
                            historical_data.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing metrics entry: {e}")
                        continue
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []
    
    async def _calculate_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trends from historical data.
        
        Args:
            historical_data: List of historical metric entries
            
        Returns:
            Dictionary with trend information
        """
        if len(historical_data) < 2:
            return {}
        
        try:
            # Get first and last data points
            first_point = historical_data[0]
            last_point = historical_data[-1]
            
            # Calculate time difference in hours
            first_time = datetime.fromisoformat(first_point['timestamp'].replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(last_point['timestamp'].replace('Z', '+00:00'))
            time_diff_hours = (last_time - first_time).total_seconds() / 3600
            
            if time_diff_hours <= 0:
                return {}
            
            # Calculate trends for key metrics
            first_counters = first_point.get('counters', {})
            last_counters = last_point.get('counters', {})
            
            trends = {}
            for key in last_counters:
                first_value = first_counters.get(key, 0)
                last_value = last_counters.get(key, 0)
                rate_per_hour = (last_value - first_value) / time_diff_hours
                trends[f"{key}_rate_per_hour"] = round(rate_per_hour, 2)
            
            # Calculate timing trends
            first_timings = first_point.get('timings', {})
            last_timings = last_point.get('timings', {})
            
            for key in last_timings:
                first_value = first_timings.get(key, 0)
                last_value = last_timings.get(key, 0)
                trends[f"{key}_trend"] = 'increasing' if last_value > first_value else 'decreasing' if last_value < first_value else 'stable'
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {}
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}
            
            # Get collection stats if database is available
            get_col = getattr(db_manager, 'get_collection', None)
            if callable(get_col):
                try:
                    # Get conversations collection stats
                    convs = get_col('conversations')
                    if convs is not None:
                        def _get_conv_stats():
                            total = convs.count_documents({})
                            active = convs.count_documents({'state': 'active'})
                            ended = convs.count_documents({'state': 'ended'})
                            pending_end = convs.count_documents({'state': 'pending_end'})
                            reopened = convs.count_documents({'state': 'reopened'})
                            
                            # Get EOC detection stats
                            eoc_detected = convs.count_documents({'eoc_confidence': {'$exists': True}})
                            
                            return {
                                'total_conversations': total,
                                'active_conversations': active,
                                'ended_conversations': ended,
                                'pending_end_conversations': pending_end,
                                'reopened_conversations': reopened,
                                'eoc_detected_conversations': eoc_detected
                            }
                        
                        conv_stats = await asyncio.to_thread(_get_conv_stats)
                        stats['conversations'] = conv_stats
                    
                    # Get messages collection stats
                    msgs = get_col('messages')
                    if msgs is not None:
                        def _get_msg_stats():
                            total = msgs.count_documents({})
                            return {'total_messages': total}
                        
                        msg_stats = await asyncio.to_thread(_get_msg_stats)
                        stats['messages'] = msg_stats
                        
                except Exception as e:
                    logger.warning(f"Error getting database stats: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance-related metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get historical metrics for performance analysis
            historical_data = await self._get_historical_metrics(hours=1)  # Last hour
            
            if not historical_data:
                return {}
            
            # Calculate performance statistics
            timing_metrics = []
            for entry in historical_data:
                timings = entry.get('timings', {})
                if timings:
                    timing_metrics.append(timings)
            
            if not timing_metrics:
                return {}
            
            # Calculate averages
            avg_timings = {}
            for key in timing_metrics[0]:
                values = [timing.get(key, 0) for timing in timing_metrics]
                avg_timings[f"avg_{key}"] = round(sum(values) / len(values), 4)
            
            # Find peaks
            peak_timings = {}
            for key in timing_metrics[0]:
                values = [timing.get(key, 0) for timing in timing_metrics]
                peak_timings[f"peak_{key}"] = round(max(values), 4)
            
            return {
                'average_timings': avg_timings,
                'peak_timings': peak_timings,
                'sample_count': len(timing_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def get_eoc_detection_stats(self) -> Dict[str, Any]:
        """
        Get detailed EOC detection statistics.
        
        Returns:
            Dictionary with EOC detection statistics
        """
        try:
            stats = {}
            
            # Get historical metrics for EOC analysis
            historical_data = await self._get_historical_metrics(hours=24)  # Last 24 hours
            
            if not historical_data:
                return {}
            
            # Count EOC detections by method
            method_counts = {}
            confidence_scores = []
            
            for entry in historical_data:
                counters = entry.get('counters', {})
                eoc_detected = counters.get('eoc_detected', 0)
                if eoc_detected > 0:
                    # This is a simplified approach - in a real system, we'd track method per detection
                    # For now, we'll use the current metrics to infer method distribution
                    pass
            
            # Get current EOC metrics
            current_metrics = get_metrics()
            counters = current_metrics.get('counters', {})
            
            stats.update({
                'total_eoc_detected': counters.get('eoc_detected', 0),
                'total_eoc_confirmed': counters.get('eoc_confirmed', 0),
                'total_summaries_sent': counters.get('summary_sent', 0),
                'total_feedback_received': counters.get('feedback_received', 0),
                'confirmation_rate': round(
                    counters.get('eoc_confirmed', 0) / max(counters.get('eoc_detected', 1), 1) * 100, 2
                ),
                'summary_success_rate': round(
                    counters.get('summary_sent', 0) / max(counters.get('eoc_confirmed', 1), 1) * 100, 2
                )
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting EOC detection stats: {e}")
            return {}
    
    async def generate_dashboard_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dashboard report.
        
        Returns:
            Dictionary with complete dashboard information
        """
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_metrics': await self.get_system_metrics(),
                'database_stats': await self.get_database_stats(),
                'performance_metrics': await self.get_performance_metrics(),
                'eoc_detection_stats': await self.get_eoc_detection_stats()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating dashboard report: {e}")
            return {}
    
    async def export_dashboard_data(self, output_file: str = None) -> str:
        """
        Export dashboard data to a JSON file.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the exported file
        """
        try:
            # Generate dashboard report
            report = await self.generate_dashboard_report()
            
            # Determine output file path
            if output_file is None:
                output_file = os.path.join(
                    os.getcwd(), 
                    f"dashboard_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                )
            
            # Export to file
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            raise

# Global dashboard instance
dashboard = MonitoringDashboard()

async def get_dashboard_report() -> Dict[str, Any]:
    """
    Get the current dashboard report.
    
    Returns:
        Dictionary with dashboard report data
    """
    return await dashboard.generate_dashboard_report()

async def export_dashboard_data(output_file: str = None) -> str:
    """
    Export current dashboard data to a file.
    
    Args:
        output_file: Optional output file path
        
    Returns:
        Path to the exported file
    """
    return await dashboard.export_dashboard_data(output_file)

async def start_dashboard_server(port: int = 8080):
    """
    Start a simple web server to serve dashboard data.
    
    Args:
        port: Port to run the server on
    """
    try:
        from aiohttp import web
        import json
        
        async def dashboard_handler(request):
            """Handler for dashboard data endpoint"""
            try:
                report = await get_dashboard_report()
                return web.json_response(report)
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)
        
        async def metrics_handler(request):
            """Handler for current metrics endpoint"""
            try:
                metrics = get_metrics()
                return web.json_response(metrics)
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)
        
        app = web.Application()
        app.router.add_get('/dashboard', dashboard_handler)
        app.router.add_get('/metrics', metrics_handler)
        app.router.add_get('/', dashboard_handler)  # Default to dashboard
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        
        logger.info(f"Dashboard server started on http://localhost:{port}")
        
        # Keep server running
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour
            
    except ImportError:
        logger.warning("aiohttp not available, dashboard server not started")
    except Exception as e:
        logger.error(f"Error starting dashboard server: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate and display a sample dashboard report
    async def main():
        report = await get_dashboard_report()
        print(json.dumps(report, indent=2, default=str))
    
    asyncio.run(main())