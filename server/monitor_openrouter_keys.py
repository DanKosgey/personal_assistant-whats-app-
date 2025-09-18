#!/usr/bin/env python3
"""
Monitoring script to track OpenRouter key usage and rotation.
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_key_usage(log_file_path=None):
    """Monitor OpenRouter key usage by parsing log files."""
    if not log_file_path:
        # Try to find the log file in common locations
        possible_paths = [
            Path(__file__).parent / "logs" / "app.log",
            Path(__file__).parent / "app.log",
            Path(__file__).parent / "server.log",
        ]
        
        log_file_path = None
        for path in possible_paths:
            if path.exists():
                log_file_path = path
                break
    
    if not log_file_path or not Path(log_file_path).exists():
        print("❌ Log file not found. Please specify a log file path.")
        print("Usage: python monitor_openrouter_keys.py /path/to/logfile.log")
        return False
    
    print(f"🔍 Monitoring OpenRouter key usage in: {log_file_path}")
    print("=" * 60)
    print("Timestamp           | Key Prefix  | Action")
    print("-" * 60)
    
    # Track key usage
    key_usage = {}
    last_position = 0
    
    try:
        while True:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                # Move to the last position we read
                f.seek(last_position)
                
                # Read new lines
                new_lines = f.readlines()
                last_position = f.tell()
                
                # Process new lines
                for line in new_lines:
                    # Look for OpenRouter key usage patterns
                    if "OPENROUTER_API_CALL:" in line:
                        timestamp = line[:23] if len(line) > 23 else "Unknown"
                        
                        # Extract key prefix if present
                        key_prefix = "Unknown"
                        if "key:" in line:
                            try:
                                # Find the key part
                                key_start = line.find("key:") + 5
                                if key_start > 4:  # Found "key:"
                                    key_part = line[key_start:].strip()
                                    # Extract first 8 characters
                                    if len(key_part) >= 8:
                                        key_prefix = key_part[:8]
                                    else:
                                        key_prefix = key_part[:4] if len(key_part) >= 4 else key_part
                            except:
                                pass
                        elif "key " in line:  # Handle different log formats
                            try:
                                # Find the key part
                                key_start = line.find("key ") + 4
                                if key_start > 3:  # Found "key "
                                    # Find the end of the key (next space or end of line)
                                    key_end = line.find(" ", key_start)
                                    if key_end == -1:  # Not found, use end of line
                                        key_end = len(line)
                                    key_part = line[key_start:key_end].strip()
                                    # Extract first 8 characters
                                    if len(key_part) >= 8:
                                        key_prefix = key_part[:8]
                                    else:
                                        key_prefix = key_part[:4] if len(key_part) >= 4 else key_part
                            except:
                                pass
                        
                        # Extract action
                        action = "Unknown action"
                        if "Starting request" in line:
                            action = "Starting request"
                        elif "Response processed" in line:
                            action = "Request successful"
                        elif "rate limit hit" in line:
                            action = "Rate limit hit!"
                        elif "API call failed" in line:
                            action = "API call failed"
                        
                        print(f"{timestamp} | {key_prefix:10} | {action}")
                        
                        # Track key usage statistics
                        if key_prefix != "Unknown":
                            if key_prefix not in key_usage:
                                key_usage[key_prefix] = {"requests": 0, "success": 0, "failures": 0}
                            
                            key_usage[key_prefix]["requests"] += 1
                            if action == "Request successful":
                                key_usage[key_prefix]["success"] += 1
                            elif "failed" in action or "limit" in action:
                                key_usage[key_prefix]["failures"] += 1
                
                # Display key usage statistics every 10 seconds
                if new_lines:
                    print("\n" + "=" * 60)
                    print("Key Usage Statistics:")
                    print("-" * 30)
                    for key, stats in key_usage.items():
                        success_rate = (stats["success"] / stats["requests"] * 100) if stats["requests"] > 0 else 0
                        print(f"Key {key}: {stats['requests']} requests, {stats['success']} success, {stats['failures']} failures ({success_rate:.1f}% success rate)")
                    print("=" * 60)
                
            # Wait before checking for new lines
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user.")
        return True
    except Exception as e:
        print(f"❌ Error monitoring log file: {e}")
        return False

def check_key_availability():
    """Check the current availability status of all OpenRouter keys."""
    try:
        from server.key_manager import KeyManager
        
        # Create a key manager
        key_manager = KeyManager()
        
        print("🔐 OpenRouter Key Availability Check")
        print("=" * 50)
        
        if not key_manager.openrouter_keys:
            print("❌ No OpenRouter keys configured")
            return False
            
        print(f"Found {len(key_manager.openrouter_keys)} OpenRouter keys:")
        
        for i, key in enumerate(key_manager.openrouter_keys):
            can_use = key.can_use()
            status = "✅ Available" if can_use else "❌ Unavailable"
            
            # Check if temporarily unavailable
            if not can_use and key._temporarily_unavailable:
                if key._unavailable_until:
                    time_remaining = key._unavailable_until - datetime.now().astimezone()
                    if time_remaining.total_seconds() > 0:
                        status = f"⏳ Unavailable for {int(time_remaining.total_seconds())} more seconds"
            
            # Show usage statistics
            usage_info = f"(Used {key._uses_this_minute}/{key.quota_per_minute} this minute)"
            if key.last_used:
                time_since_last_use = datetime.now().astimezone() - key.last_used
                usage_info += f" Last used {int(time_since_last_use.total_seconds())}s ago"
            
            print(f"  {i+1}. {key.key[:12]}...{key.key[-4:]} - {status} {usage_info}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking key availability: {e}")
        return False

def main():
    """Main monitoring function."""
    import sys
    
    print("OpenRouter Key Rotation Monitor")
    print("=" * 40)
    
    # Check key availability
    check_key_availability()
    
    print("\n" + "=" * 40)
    
    # If a log file path was provided, monitor it
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
        print(f"Starting log monitoring for: {log_file_path}")
        monitor_key_usage(log_file_path)
    else:
        print("To monitor key usage in real-time, provide a log file path:")
        print("Usage: python monitor_openrouter_keys.py /path/to/logfile.log")
        print("\n💡 Tip: Run your WhatsApp bot and then use this script to monitor key rotation!")

if __name__ == "__main__":
    main()