import gspread
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from google.oauth2.service_account import Credentials
import os
import logging

logger = logging.getLogger(__name__)

class GoogleSheetsLogger:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_SHEETS_API_KEY")
        self.spreadsheet_id = None
        self.spreadsheet = None
        self.worksheet = None
        self.client = None
        self.service_account_email = None
        # Initialize client after fields are set
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Google Sheets client with API key authentication"""
        try:
            # Try to use service account credentials. Prefer explicit path from
            # environment variable `GOOGLE_SHEETS_CREDENTIALS` if provided.
            creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                self.client = gspread.service_account(filename=creds_path)
                logger.info(f"Google Sheets client initialized using {creds_path}")
                # Try to read client email from the service account JSON so the
                # user can share spreadsheets with this account easily.
                try:
                    with open(creds_path, 'r', encoding='utf-8') as fh:
                        info = json.load(fh)
                        self.service_account_email = info.get('client_email')
                        if self.service_account_email:
                            logger.info(f"Google Sheets service account email: {self.service_account_email}")
                except Exception as _e:
                    logger.debug(f"Could not read service account email from {creds_path}: {_e}")
            else:
                # Fallback to gspread default lookup (e.g. GOOGLE_APPLICATION_CREDENTIALS)
                try:
                    self.client = gspread.service_account()
                    logger.info("Google Sheets client initialized (default lookup)")
                    # If the default lookup succeeded, try to read credentials path
                    # from environment variable(s) and extract client_email for convenience.
                    default_creds = os.getenv('GOOGLE_SHEETS_CREDENTIALS') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                    if default_creds and os.path.exists(default_creds):
                        try:
                            with open(default_creds, 'r', encoding='utf-8') as fh:
                                info = json.load(fh)
                                self.service_account_email = info.get('client_email')
                                if self.service_account_email:
                                    logger.info(f"Google Sheets service account email: {self.service_account_email}")
                        except Exception as _e:
                            logger.debug(f"Could not read service account email from {default_creds}: {_e}")
                except Exception:
                    raise
        except Exception as e:
            logger.warning(f"Failed to initialize Google Sheets client: {e}")
            logger.info("Google Sheets logging will be disabled. To enable:")
            logger.info("1. Create Google Cloud project")
            logger.info("2. Enable Google Sheets API")
            logger.info("3. Create service account and download JSON")
            logger.info("4. Save as ~/AppData/Roaming/gspread/service_account.json")
            # Fallback: We'll use local logging instead
            self.client = None
    
    def setup_spreadsheet(self, spreadsheet_name="WhatsApp AI Agent Logs"):
        """Setup or create the logging spreadsheet"""
        try:
            if not self.client:
                logger.warning("Google Sheets client not available, using local logging")
                return False
            # If an explicit spreadsheet id is configured, prefer opening that
            env_sheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
            if env_sheet_id:
                try:
                    spreadsheet = self.client.open_by_key(env_sheet_id)
                    logger.info(f"Opened spreadsheet by id from env: {env_sheet_id}")
                except Exception as e:
                    logger.error(f"Failed to open spreadsheet by id {env_sheet_id}: {e}")
                    return False
            else:
                try:
                    # Try to open existing spreadsheet by name
                    spreadsheet = self.client.open(spreadsheet_name)
                    logger.info(f"Opened existing spreadsheet: {spreadsheet_name}")
                except gspread.SpreadsheetNotFound:
                    # Create new spreadsheet (requires Drive API enabled)
                    try:
                        spreadsheet = self.client.create(spreadsheet_name)
                        logger.info(f"Created new spreadsheet: {spreadsheet_name}")
                    except Exception as e:
                        # Provide actionable guidance if Drive API is disabled
                        msg = str(e)
                        logger.error(f"Error creating spreadsheet: {msg}")
                        if 'drive.googleapis.com' in msg or 'SERVICE_DISABLED' in msg or 'Google Drive API' in msg:
                            logger.error("Google Drive API appears to be disabled for the service account project.")
                            logger.error("Enable the Drive API in your Google Cloud project or provide an existing spreadsheet id via GOOGLE_SHEETS_SPREADSHEET_ID.")
                        return False
            
            self.spreadsheet = spreadsheet
            self.spreadsheet_id = spreadsheet.id
            
            # Setup or get the main worksheet
            try:
                self.worksheet = spreadsheet.worksheet("Conversation_Logs")
            except gspread.WorksheetNotFound:
                self.worksheet = spreadsheet.add_worksheet(
                    title="Conversation_Logs",
                    rows=1000,
                    cols=15
                )
                
                # Add headers
                headers = [
                    "Timestamp", "Date", "Time", "Phone_Number", "Contact_Name",
                    "Is_Known_Contact", "Message_Count", "Priority", "Category",
                    "Conversation_Summary", "Duration_Minutes", "AI_Response_Count",
                    "First_Message", "Last_Message", "Status"
                ]
                self.worksheet.append_row(headers)
                logger.info("Added headers to new worksheet")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up spreadsheet: {e}")
            return False
    
    def log_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """Log conversation data to Google Sheets"""
        try:
            if not self.worksheet:
                if not self.setup_spreadsheet():
                    return self._log_locally(conversation_data)
            
            # Prepare row data
            timestamp = datetime.now(timezone.utc)
            row_data = [
                timestamp.isoformat(),
                timestamp.strftime("%Y-%m-%d"),
                timestamp.strftime("%H:%M:%S"),
                conversation_data.get("phone_number", ""),
                conversation_data.get("contact_name", "Unknown"),
                conversation_data.get("is_known_contact", False),
                conversation_data.get("message_count", 0),
                conversation_data.get("priority", "MEDIUM"),
                conversation_data.get("category", "general"),
                conversation_data.get("summary", ""),
                conversation_data.get("duration_minutes", 0),
                conversation_data.get("ai_response_count", 0),
                conversation_data.get("first_message", "")[:100],  # Limit length
                conversation_data.get("last_message", "")[:100],   # Limit length
                conversation_data.get("status", "completed")
            ]
            
            # Append to sheet
            self.worksheet.append_row(row_data)
            logger.info(f"Logged conversation for {conversation_data.get('phone_number')} to Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"Error logging to Google Sheets: {e}")
            # Fallback to local logging
            return self._log_locally(conversation_data)
    
    def log_conversation_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Log conversation summary to Google Sheets"""
        # Convert summary data to conversation format for logging
        conversation_data = {
            "phone_number": summary_data.get("phone_number", ""),
            "contact_name": summary_data.get("contact_name", "Unknown"),
            "is_known_contact": True,  # Assume known if we have a summary
            "message_count": summary_data.get("message_count", 0),
            "priority": summary_data.get("priority", "MEDIUM"),
            "category": summary_data.get("category", "general"),
            "summary": summary_data.get("summary", ""),
            "duration_minutes": 0,  # Could calculate from timestamps
            "ai_response_count": summary_data.get("message_count", 0) // 2,  # Estimate
            "first_message": "",
            "last_message": "",
            "status": "summarized"
        }
        return self.log_conversation(conversation_data)
    
    def _log_locally(self, conversation_data: Dict[str, Any]) -> bool:
        """Fallback local logging when Google Sheets is not available"""
        try:
            # Use Windows-compatible temp directory
            import tempfile
            log_file = os.path.join(tempfile.gettempdir(), "whatsapp_agent_conversations.jsonl")
            
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **conversation_data
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            logger.info(f"Logged conversation locally for {conversation_data.get('phone_number')}")
            return True
            
        except Exception as e:
            logger.error(f"Error in local logging: {e}")
            return False
    
    def get_conversation_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get conversation statistics from the last N days"""
        try:
            if not self.worksheet:
                return self._get_local_stats(days)
            
            # Get all records
            records = self.worksheet.get_all_records()
            
            # Filter by date range
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).date()
            recent_records = []
            
            for record in records:
                try:
                    record_date = datetime.fromisoformat(record['Timestamp']).date()
                    if record_date >= cutoff_date:
                        recent_records.append(record)
                except:
                    continue
            
            # Calculate stats
            total_conversations = len(recent_records)
            priority_breakdown = {}
            category_breakdown = {}
            
            for record in recent_records:
                priority = record.get('Priority', 'MEDIUM')
                category = record.get('Category', 'general')
                
                priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                category_breakdown[category] = category_breakdown.get(category, 0) + 1
            
            return {
                "total_conversations": total_conversations,
                "priority_breakdown": priority_breakdown,
                "category_breakdown": category_breakdown,
                "period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting stats from Google Sheets: {e}")
            return self._get_local_stats(days)
    
    def _get_local_stats(self, days: int) -> Dict[str, Any]:
        """Get stats from local log file"""
        try:
            # Use Windows-compatible temp directory
            import tempfile
            log_file = os.path.join(tempfile.gettempdir(), "whatsapp_agent_conversations.jsonl")
            
            if not os.path.exists(log_file):
                return {"total_conversations": 0, "priority_breakdown": {}, "category_breakdown": {}}
            
            from datetime import timedelta
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            total_conversations = 0
            priority_breakdown = {}
            category_breakdown = {}
            
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_time = datetime.fromisoformat(record['timestamp'])
                        
                        if record_time >= cutoff_date:
                            total_conversations += 1
                            
                            priority = record.get('priority', 'MEDIUM')
                            category = record.get('category', 'general')
                            
                            priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                            category_breakdown[category] = category_breakdown.get(category, 0) + 1
                    except:
                        continue
            
            return {
                "total_conversations": total_conversations,
                "priority_breakdown": priority_breakdown,
                "category_breakdown": category_breakdown,
                "period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting local stats: {e}")
            return {"total_conversations": 0, "priority_breakdown": {}, "category_breakdown": {}}

# Global instance
sheets_logger = GoogleSheetsLogger()