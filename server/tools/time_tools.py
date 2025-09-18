import logging
from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class LLMTimeTools:
    """Enhanced time tools with advanced time intelligence for AI agents"""
    
    def __init__(self):
        # Common timezone mappings for user-friendly names
        self.timezone_mappings = {
            "UTC": "UTC",
            "GMT": "UTC",
            "EST": "America/New_York",
            "PST": "America/Los_Angeles",
            "MST": "America/Denver",
            "CST": "America/Chicago",
            "IST": "Asia/Kolkata",
            "BST": "Europe/London",
            "CET": "Europe/Paris",
            "JST": "Asia/Tokyo",
            "AEST": "Australia/Sydney",
            "HST": "Pacific/Honolulu"
        }
    
    def get_current_time(self, timezone: Optional[str] = None) -> Dict[str, Any]:
        """Get current time with advanced context information"""
        try:
            # Use provided timezone or default to UTC
            if timezone:
                # Handle common timezone abbreviations
                if timezone.upper() in self.timezone_mappings:
                    tz_name = self.timezone_mappings[timezone.upper()]
                else:
                    tz_name = timezone
                
                try:
                    tz = pytz.timezone(tz_name)
                except pytz.exceptions.UnknownTimeZoneError:
                    # Fallback to UTC if timezone is invalid
                    tz = pytz.UTC
            else:
                tz = pytz.UTC
            
            # Get current time in specified timezone
            now = datetime.now(tz)
            
            # Determine time of day context
            hour = now.hour
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"
            
            # Determine day context
            day_of_week = now.strftime("%A")
            is_weekend = now.weekday() >= 5
            
            # Determine season (Northern Hemisphere approximation)
            month = now.month
            if month in [12, 1, 2]:
                season = "winter"
            elif month in [3, 4, 5]:
                season = "spring"
            elif month in [6, 7, 8]:
                season = "summer"
            else:
                season = "autumn"
            
            return {
                "formatted": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "iso": now.isoformat(),
                "timezone": str(tz),
                "hour": hour,
                "minute": now.minute,
                "second": now.second,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "time_of_day": time_of_day,
                "season": season,
                "timestamp": now.timestamp()
            }
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            # Fallback to basic UTC time
            now = datetime.utcnow()
            return {
                "formatted": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "iso": now.isoformat() + "Z",
                "timezone": "UTC",
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "day_of_week": now.strftime("%A"),
                "is_weekend": now.weekday() >= 5,
                "time_of_day": "day" if 6 <= now.hour < 18 else "night",
                "season": "unknown",
                "timestamp": now.timestamp()
            }
    
    def get_time_context_for_user(self, user_timezone: Optional[str] = None, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get enhanced time context tailored for user interaction"""
        time_info = self.get_current_time(user_timezone)
        
        # Add user-specific time intelligence
        context = {
            "current_time": time_info,
            "greeting_time": time_info["time_of_day"],
            "appropriate_greeting": self._get_appropriate_greeting(time_info),
            "time_based_suggestions": self._get_time_based_suggestions(time_info, user_preferences),
            "cultural_context": self._get_cultural_context(time_info)
        }
        
        return context
    
    def _get_appropriate_greeting(self, time_info: Dict[str, Any]) -> str:
        """Get appropriate greeting based on time of day"""
        time_of_day = time_info["time_of_day"]
        is_weekend = time_info["is_weekend"]
        
        if time_of_day == "morning":
            if is_weekend:
                return "Good morning! Hope you're enjoying your weekend."
            else:
                return "Good morning! Hope you're having a productive day."
        elif time_of_day == "afternoon":
            if is_weekend:
                return "Good afternoon! Hope you're having a relaxing weekend."
            else:
                return "Good afternoon! How's your day going?"
        elif time_of_day == "evening":
            return "Good evening! How has your day been?"
        else:  # night
            return "Hello! It's quite late. Hope you're doing well."
    
    def _get_time_based_suggestions(self, time_info: Dict[str, Any], user_preferences: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get time-based suggestions for user interaction"""
        suggestions = []
        hour = time_info["hour"]
        time_of_day = time_info["time_of_day"]
        is_weekend = time_info["is_weekend"]
        
        # Time-based suggestions
        if 6 <= hour < 9:
            suggestions.append("Perfect time for planning your day ahead")
        elif 9 <= hour < 12:
            suggestions.append("Great time for focused work or meetings")
        elif 12 <= hour < 14:
            suggestions.append("Time for a break or lunch")
        elif 14 <= hour < 17:
            suggestions.append("Good time for afternoon tasks or follow-ups")
        elif 17 <= hour < 19:
            suggestions.append("Time to wrap up work and plan for tomorrow")
        elif 19 <= hour < 22:
            suggestions.append("Evening is great for reflection or planning")
        else:
            suggestions.append("Late night work? Don't forget to rest")
        
        # Weekend vs weekday suggestions
        if is_weekend:
            if time_of_day == "morning":
                suggestions.append("Weekend mornings are perfect for personal time")
            elif time_of_day == "afternoon":
                suggestions.append("Great time for hobbies or relaxation")
        else:
            if time_of_day == "morning":
                suggestions.append("Productive morning for work tasks")
            elif time_of_day == "afternoon":
                suggestions.append("Mid-day energy boost for important tasks")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _get_cultural_context(self, time_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get cultural context based on time and season"""
        return {
            "seasonal_context": f"It's {time_info['season']} season",
            "day_context": "weekend" if time_info["is_weekend"] else "weekday",
            "time_context": time_info["time_of_day"]
        }
    
    def format_time_for_user(self, timestamp: float, user_timezone: Optional[str] = None) -> str:
        """Format timestamp in user-friendly way with timezone context"""
        try:
            if user_timezone:
                if user_timezone.upper() in self.timezone_mappings:
                    tz_name = self.timezone_mappings[user_timezone.upper()]
                else:
                    tz_name = user_timezone
                tz = pytz.timezone(tz_name)
            else:
                tz = pytz.UTC
            
            dt = datetime.fromtimestamp(timestamp, tz)
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception as e:
            logger.error(f"Error formatting time: {e}")
            # Fallback to basic formatting
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")