import os
import asyncio
import json
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, NoReturn
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

# MongoDB imports
import pymongo
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, field_validator  # Fixed: validator -> field_validator
import httpx
# Import Gemini AI but avoid using any private APIs
from typing import Any
from typing import Any, Protocol

class GenerativeAI(Protocol):
    def configure(self, api_key: str) -> None: ...
    def GenerativeModel(self, model_name: str) -> Any: ...

class FallbackGenerativeAI:
    def configure(self, api_key: str) -> None:
        pass
    
    def GenerativeModel(self, model_name: str) -> Any:
        return None

try:
    import google.generativeai as genai  # type: ignore
    genai: GenerativeAI
except ImportError:
    genai = FallbackGenerativeAI()
# Do not import GenerativeModel directly as it's not exported
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import logging
import redis
from cachetools import TTLCache
import hashlib
import re
from urllib.parse import urlparse
import aiofiles
import aiocron
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# OpenRouter fallback client
from openrouter_client import OpenRouterClient
from persona_manager import PersonaManager

# Advanced logging configuration
class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for better log visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['ENDC'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)

# Setup advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whatsapp_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Apply colored formatter to console handler
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and hasattr(handler.stream, 'isatty') and handler.stream.isatty():
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Enums for better type safety
class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    LOCATION = "location"
    CONTACT = "contact"
    STICKER = "sticker"

class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SPAM = "SPAM"

class Category(str, Enum):
    BUSINESS = "business"
    INVESTMENT = "investment"
    SUPPORT = "support"
    PERSONAL = "personal"
    NETWORKING = "networking"
    SALES = "sales"
    MEDIA = "media"
    SPAM = "spam"
    GENERAL = "general"

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    WAITING = "waiting"
    CLOSED = "closed"
    ARCHIVED = "archived"

# Fixed QueryBase class
class QueryBase(dict):
    """Base class for MongoDB query dictionaries with proper type hints."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_types()
        
    def _validate_types(self):
        """Validate that all values have proper types for MongoDB."""
        for key, value in self.items():
            if not isinstance(value, (str, int, float, bool, datetime, dict, list, type(None))):
                logger.warning(f"Potentially unsupported type for MongoDB: {type(value)} for key '{key}'")
        
    def __setitem__(self, key: str, value: Any):
        """Override to ensure type safety."""
        if isinstance(value, (str, int, float, bool, datetime, dict, list, type(None))):
            super().__setitem__(key, value)
        else:
            raise TypeError(f"Value of type {type(value)} is not supported in MongoDB queries")

# Advanced Configuration Management
@dataclass
class AgentConfig:
    # Core API Keys
    mongo_url: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    whatsapp_access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    whatsapp_phone_number_id: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    
    # Multiple Gemini API keys support
    @property
    def gemini_api_keys(self) -> List[str]:
        """Get list of Gemini API keys from environment"""
        # Get primary key
        # Collect any environment variables that start with GEMINI_API_KEY
        keys = []  # list of (name, value)
        for name, val in os.environ.items():
            if name.startswith("GEMINI_API_KEY") and val:
                keys.append((name, val))

        if not keys:
            return []

        # Sort keys so that:
        # - GEMINI_API_KEY (no suffix) comes first
        # - GEMINI_API_KEY_<n> are sorted by n
        # - any other GEMINI_API_KEY* names come after, sorted lexicographically
        import re as _re
        def _sort_key(item):
            nm = item[0]
            if nm == "GEMINI_API_KEY":
                return (0, 0, nm)
            m = _re.match(r"GEMINI_API_KEY_(\d+)$", nm)
            if m:
                return (1, int(m.group(1)), nm)
            return (2, 0, nm)

        keys.sort(key=_sort_key)
        return [val for _, val in keys]
    webhook_verify_token: str = os.getenv("WEBHOOK_VERIFY_TOKEN", "whatsapp_webhook_2025")
    database_name: str = os.getenv("DATABASE_NAME", "whatsapp_agent_v2")
    
    # Redis Configuration
    redis_url: Optional[str] = os.getenv("REDIS_URL", None)
    
    # Agent Personality & Behavior
    user_personality: str = os.getenv("USER_PERSONALITY", 
        "You are a friendly and empathetic assistant who helps manage communications. "
        "While representing a successful entrepreneur, you focus on building genuine connections "
        "and understanding people's needs through natural conversation.")
    
    # Advanced Features
    enable_sentiment_analysis: bool = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true"
    enable_smart_routing: bool = os.getenv("ENABLE_SMART_ROUTING", "true").lower() == "true"
    enable_auto_summary: bool = os.getenv("ENABLE_AUTO_SUMMARY", "true").lower() == "true"
    enable_learning_mode: bool = os.getenv("ENABLE_LEARNING_MODE", "true").lower() == "true"
    
    # Rate Limiting
    max_messages_per_minute: int = int(os.getenv("MAX_MESSAGES_PER_MINUTE", "10"))
    max_requests_per_ip: int = int(os.getenv("MAX_REQUESTS_PER_IP", "100"))
    
    # Business Hours
    business_start_hour: int = int(os.getenv("BUSINESS_START_HOUR", "9"))
    business_end_hour: int = int(os.getenv("BUSINESS_END_HOUR", "18"))
    timezone_offset: int = int(os.getenv("TIMEZONE_OFFSET", "0"))
    
    # Owner Notifications
    owner_whatsapp_number: Optional[str] = os.getenv("OWNER_WHATSAPP_NUMBER")
    notification_threshold: int = int(os.getenv("NOTIFICATION_THRESHOLD", "5"))
    # Testing/dev: disable real WhatsApp API sends when true
    disable_whatsapp_sends: bool = os.getenv("DISABLE_WHATSAPP_SENDS", "false").lower() == "true"
    # Production WhatsApp settings
    # Comma-separated list of allowed recipient phone numbers (E.164) for safety; empty means allow all
    whatsapp_allowed_recipients: List[str] = field(default_factory=lambda: [x.strip() for x in os.getenv("WHATSAPP_ALLOWED_RECIPIENTS", "").split(",") if x.strip()])
    # Number of retries for transient WhatsApp send failures
    whatsapp_send_max_retries: int = int(os.getenv("WHATSAPP_SEND_MAX_RETRIES", "3"))
    # Base backoff seconds (exponential backoff multiplier)
    whatsapp_send_backoff_base: float = float(os.getenv("WHATSAPP_SEND_BACKOFF_BASE", "0.5"))
    
    # Google Sheets Integration
    google_sheets_credentials: Optional[str] = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
    google_sheets_spreadsheet_id: Optional[str] = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
    # OpenRouter fallback configuration
    enable_openrouter_fallback: bool = os.getenv("ENABLE_OPENROUTER_FALLBACK", "true").lower() == "true"
    openrouter_model: Optional[str] = os.getenv("OPENROUTER_MODEL_ID")
    openrouter_timeout: int = int(os.getenv("OPENROUTER_TIMEOUT", "20"))
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.whatsapp_access_token:
            logger.warning("WhatsApp access token not provided")
        # Warn if no Gemini API keys were provided
        if not self.gemini_api_keys:
            logger.warning("Gemini API key(s) not provided")

config = AgentConfig()

# Persona manager: allows swapping agent persona by changing a JSON file or env var
persona_manager = PersonaManager(default_prompt=config.user_personality)

# Advanced Pydantic Models
class ContactInfo(BaseModel):
    phone_number: str
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    is_known: bool = False
    is_vip: bool = False
    priority_level: Priority = Priority.MEDIUM
    interaction_count: int = 0
    satisfaction_score: float = 0.0
    last_interaction: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    blocked: bool = False

    @field_validator('phone_number')  # Fixed: validator -> field_validator
    @classmethod
    def validate_phone_number(cls, v):
        import re
        if not re.match(r'^\+?[\d\s\-\(\)]+$', v):
            raise ValueError("Invalid phone number format")
        return v
    
    class Config:
        use_enum_values = True

class MessageAnalysis(BaseModel):
    sentiment: float = Field(..., ge=-1, le=1, description="Sentiment score from -1 (negative) to 1 (positive)")
    emotions: Dict[str, float] = Field(default_factory=dict)
    intent: str = ""
    urgency: float = Field(0.0, ge=0, le=1)
    business_relevance: float = Field(0.0, ge=0, le=1)
    language: str = "en"
    contains_pii: bool = False
    spam_score: float = Field(0.0, ge=0, le=1)
    # Style hints the AI may return or we set (language/formality/verbosity)
    style: Dict[str, Any] = Field(default_factory=dict)

class EnhancedMessage(BaseModel):
    message_id: str
    conversation_id: str
    phone_number: str
    message_type: MessageType
    content: str
    timestamp: datetime
    is_from_user: bool
    ai_response: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    category: Category = Category.GENERAL
    analysis: Optional[MessageAnalysis] = None
    processing_time_ms: int = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    # Telemetry about which provider generated the reply and send status
    provider_used: Optional[str] = None
    send_status: str = "pending"
    
    class Config:
        use_enum_values = True

class ConversationSummary(BaseModel):
    conversation_id: str
    contact_info: ContactInfo
    message_count: int
    priority: Priority
    category: Category
    summary: str
    key_topics: List[str]
    action_items: List[str]
    sentiment_trend: List[float]
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    satisfaction_score: float
    resolution_status: str
    
    class Config:
        use_enum_values = True

# MongoDB type checking helpers
def verify_collection_methods(collection: Any) -> bool:
    """Verify collection has required MongoDB methods"""
    required_methods = ['find_one', 'find', 'insert_one', 'update_one']
    try:
        return all(callable(getattr(collection, method, None)) for method in required_methods)
    except:
        return False

def is_valid_collection(collection: Any) -> bool:
    """Safe check for MongoDB collection validity"""
    if collection is None:
        return False
    try:
        # Check for Collection type and required methods
        is_collection = isinstance(collection, Collection)
        has_methods = verify_collection_methods(collection)
        return is_collection and has_methods
    except Exception:
        return False

def get_collection(collection: Any) -> Optional[Collection]:
    """Safe get collection with type checking"""
    return collection if is_valid_collection(collection) else None

# Advanced Database Manager - Fixed version
class DatabaseManager:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None  # Type: Optional[pymongo.database.Database]
        self.collections: Dict[str, Optional[Collection]] = {}
        self.redis_client = None
        
    def get_collection(self, name: str) -> Optional[Collection]:
        """Get a collection if it exists and is valid"""
        try:
            # Basic validation
            if name not in self.collections:
                return None
                
            # Get collection with type checking
            collection = self.collections.get(name)
            
            # Verify collection is valid
            if not is_valid_collection(collection):
                return None
                
            return collection
            
        except Exception as e:
            logger.debug(f"Collection access error: {e}")
            return None
        
    def has_collection(self, name: str) -> bool:
        """Check if a collection exists"""
        try:
            collection = self.get_collection(name)
            return collection is not None
        except Exception:
            return False
        
    async def initialize(self, mongo_url: Optional[str] = None, database_name: Optional[str] = None, redis_url: Optional[str] = None):
        """Initialize database connections with proper error handling"""
        # Use provided URLs or fall back to config
        mongo_url = mongo_url if mongo_url is not None else config.mongo_url
        database_name = database_name if database_name is not None else config.database_name
        redis_url = redis_url if redis_url is not None else config.redis_url
        
        # Initialize with defaults
        self.client = None
        self.db = None
        self.collections = {}

        if not mongo_url:
            logger.warning("MongoDB URL not configured")
            return

        try:
            # MongoDB connection with advanced options
            self.client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=100,
                retryWrites=True
            )
            
            # Test connection
            await asyncio.to_thread(self.client.admin.command, 'ping')
            self.db = self.client[database_name]
            
            # Initialize collections with type safety
            try:
                for collection_name in ['conversations', 'contacts', 'messages', 'analytics', 'feedback', 'templates', 'audit_log']:
                    try:
                        if hasattr(self.db, collection_name):
                            collection = getattr(self.db, collection_name)
                            if verify_collection_methods(collection):
                                self.collections[collection_name] = collection
                            else:
                                logger.warning(f"Collection {collection_name} missing required methods")
                        else:
                            logger.warning(f"Collection {collection_name} not found in database")
                    except Exception as e:
                        logger.error(f"Error initializing collection {collection_name}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error accessing database: {e}")            # Create indexes for better performance
            await self._create_indexes()
            
            logger.info("MongoDB connected successfully")
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            # Don't raise in production, continue with limited functionality
            self.client = None
            self.db = None
            self.collections = {}
        
        # Only attempt Redis connection if URL is configured
        if redis_url:
            try:
                # Redis connection for caching and rate limiting
                import redis.asyncio as aioredis
                self.redis_client = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20
                )
                
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("Redis connected successfully")
                
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to memory cache.")
                self.redis_client = None
        else:
            logger.info("Redis URL not configured. Using memory cache only.")
            self.redis_client = None
    
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            if self.db is None or self.client is None:
                logger.warning("Database not initialized - skipping index creation")
                return

            # Messages indexes
            messages_collection = self.collections.get('messages')
            if messages_collection is not None:
                try:
                    await asyncio.to_thread(
                        messages_collection.create_index,
                        [("conversation_id", 1), ("timestamp", -1)]
                    )
                    await asyncio.to_thread(
                        messages_collection.create_index,
                        [("phone_number", 1), ("timestamp", -1)]
                    )
                except Exception as e:
                    logger.error(f"Error creating messages indexes: {e}")
            
            # Conversations indexes
            conversations_collection = self.collections.get('conversations')
            if conversations_collection is not None:
                try:
                    await asyncio.to_thread(
                        conversations_collection.create_index,
                        [("phone_number", 1), ("start_time", -1)]
                    )
                except Exception as e:
                    logger.error(f"Error creating conversations index: {e}")
            
            # Contacts indexes
            contacts_collection = self.collections.get('contacts')
            if contacts_collection is not None:
                try:
                    await asyncio.to_thread(
                        contacts_collection.create_index,
                        [("phone_number", 1)],
                        unique=True
                    )
                except Exception as e:
                    logger.error(f"Error creating contacts index: {e}")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    async def close(self):
        """Close database connections"""
        if self.client:
            self.client.close()
        if self.redis_client:
            await self.redis_client.close()

# Initialize database manager
db_manager = DatabaseManager()

# Advanced Caching System
class CacheManager:
    def __init__(self):
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)
        self.message_dedup = TTLCache(maxsize=5000, ttl=3600)
        self.rate_limits = TTLCache(maxsize=1000, ttl=60)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis first, then memory)"""
        try:
            if db_manager.redis_client:
                value = await db_manager.redis_client.get(key)
                if value:
                    return json.loads(value)
        except Exception as e:
            logger.debug(f"Redis get failed: {e}")
        
        return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        try:
            if db_manager.redis_client:
                await db_manager.redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.debug(f"Redis set failed: {e}")
        
        self.memory_cache[key] = value
    
    def is_duplicate_message(self, message_id: str) -> bool:
        """Check if message is duplicate"""
        if message_id in self.message_dedup:
            return True
        self.message_dedup[message_id] = True
        return False
    
    def check_rate_limit(self, key: str, limit: int) -> bool:
        """Check rate limit"""
        current = self.rate_limits.get(key, 0)
        if current >= limit:
            return False
        self.rate_limits[key] = current + 1
        return True

cache_manager = CacheManager()

# Advanced AI Handler with Multiple Capabilities
class AdvancedAIHandler:
    def __init__(self):
        self.models = []
        self.current_model_index = 0
        self.ai_available = False
        self.last_key_rotation = datetime.now()
        self.key_attempt_counts = {}  # Track 429 errors per key
        self.next_retry_time: Optional[datetime] = None
        # Prevent sending duplicate owner notifications for the same exhaustion event
        self.owner_notified: bool = False
        # Internal task handle for re-enabling AI after an ETA
        self._reenable_task = None
        self.init_ai()

    def init_ai(self):
        """Initialize AI models with multiple API keys and proper error handling"""
        api_keys = config.gemini_api_keys
        
        if not api_keys:
            logger.warning("No Gemini API keys configured - using fallback responses")
            return

        try:
            if not genai:
                logger.error("Failed to import google.generativeai")
                return

            # Initialize models for each API key
            for key in api_keys:
                try:
                    genai.configure(api_key=key)
                    model = genai.GenerativeModel('gemini-1.5-flash')  # type: ignore

                    # Test model initialization
                    if not hasattr(model, 'generate_content'):
                        logger.warning(f"Model initialization failed for one API key - skipping")
                        continue

                    self.models.append({
                        'model': model,
                        'api_key': key,
                        'last_error': None,
                        'error_count': 0,
                        'last_success': datetime.now()
                    })

                except Exception as e:
                    logger.warning(f"Failed to initialize model for one API key: {e}")
                    continue

            if self.models:
                self.ai_available = True
                logger.info(f"Gemini AI initialized successfully with {len(self.models)} API keys")
            else:
                logger.warning("No valid Gemini models could be initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            logger.warning("Using fallback responses")
        
        # Initialize TF-IDF for similarity analysis
        try:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.conversation_vectors = {}
        except Exception as e:
            logger.warning(f"TF-IDF initialization failed: {e}")
            self.vectorizer = None
            self.conversation_vectors = {}
        
        # Personality and system prompts
        self.system_prompt = self._build_system_prompt()
        
        # Quick response templates
        # Bilingual quick response templates (English and Swahili)
        self.quick_responses = {
            "greeting": {
                "en": [
                    "Hello! I'm here to help. Please let me know your name and how I can assist you today.",
                    "Hi there! Thanks for reaching out. Could you please introduce yourself and tell me what you need?",
                    "Good day! I appreciate you contacting me. Please share your name and the purpose of your message."
                ],
                "sw": [
                    "Habari! Niko hapa kukusaidia. Tafadhali nipe jina lako na nikupe msaada gani leo.",
                    "Hujambo! Asante kwa kuwasiliana. Tafadhali jitambulisha na niambie unahitaji nini.",
                    "Siku njema! Tafadhali shiriki jina lako na sababu ya kuwasiliana nami."]
            },
            "goodbye": {
                "en": ["Thank you for your time. Have a wonderful day!","It was great speaking with you. Feel free to reach out anytime!","Goodbye! I'm always here if you need assistance."],
                "sw": ["Asante kwa muda wako. Uwe na siku njema!","Ilikuwa vizuri kuzungumza nawe. Wasiliana tena wakati wowote!","Kwaheri! Niko hapa kila wakati ukihitaji msaada."]
            },
            "unknown_contact": {
                "en": [
                    "I don't believe we've been introduced. Could you please tell me your name and the reason for your contact?",
                    "Thanks for reaching out! I'd like to know who I'm speaking with. Please share your name and what you need.",
                    "Hello! For better assistance, could you please introduce yourself and explain how I can help?"
                ],
                "sw": [
                    "Sidhani tumekutana kabla. Tafadhali nipe jina lako na sababu ya kuwasiliana.",
                    "Asante kwa kuwasiliana! Ningependa kujua ninayezungumza naye. Tafadhali nitaje na unahitaji nini.",
                    "Habari! Ili niweze kukusaidia vyema, tafadhali jitenulishe na eleza unahitaji nini."
                ]
            },
            "fallback": {
                "en": ["Thank you for your message. I'll get back to you shortly.","I've received your message and will respond as soon as possible.","Thanks for reaching out. Let me process your request."],
                "sw": ["Asante kwa ujumbe wako. Nitarudia hivi karibuni.","Nimepokea ujumbe wako na nitajibu haraka iwezekanavyo.","Asante kwa kuwasiliana. Niruhusu niweke taratibu za kujibu."]
            }
        }
        # Lightweight Swahili common words set used to validate fluency and spelling.
        # This is intentionally small and conservative — it helps detect clearly non-Swahili outputs.
        self.swahili_common_words = set([
            'habari','hujambo','mambo','asante','tafadhali','karibu','ndio','hapana','sawa','nzuri',
            'siku','leo','jana','kesho','mimi','wewe','yeye','sisi','nyinyi','wao','kuhusu','kutoka',
            'kwa','na','ya','wa','ni','kwao','kama','lakini','naweza','takwimu','maelezo','msaada',
            'huduma','jina','tafuta','tafadhali','nitakusaidia','asilimia','karibuni','samahani','salamu'
        ])
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt"""
        # Use the configured persona's system prompt when available. persona_manager defaults
        # to config.user_personality if no persona files or selection exist.
        persona_prompt = persona_manager.get_system_prompt() if 'persona_manager' in globals() else config.user_personality
  
        return f"""
{persona_prompt}

You are a friendly and intelligent assistant who manages communication. Your role is to:

1. Be natural and conversational in your tone
2. Show genuine interest in people's messages
3. Be helpful and understanding
4. Remember context and details from conversations
5. Use appropriate language based on the situation

Language behavior:
- Detect the sender's language for each incoming message. Support English and Swahili primarily.
- If the sender writes in Swahili (Kiswahili), respond in fluent, natural Swahili.
- If the sender writes in English, respond in clear, natural English.
- If the sender mixes both languages, prefer the language of the last user message but you may code-switch naturally when appropriate.
- If you cannot determine the language, default to English and be polite.

Guidelines:
- When meeting someone new, be welcoming and friendly while getting to know them
- Build rapport and maintain relationships with regular contacts
- Show empathy and understanding in your responses
- Be proactive in helping solve problems or address concerns
- Keep conversations flowing naturally while gathering necessary information
- Be flexible and adaptive to different conversation styles
- While protecting private information, try to be as helpful as possible

For Business Matters:
- Listen carefully to business proposals and ideas
- Help coordinate meetings and follow-ups effectively
- Show appropriate enthusiasm for opportunities while maintaining professionalism
- Gather relevant details naturally through conversation
- Be clear and transparent about next steps

Remember: You're having real conversations with real people. Be genuine, helpful, and human in your interactions.
        """
    
    async def _try_api_call(self, prompt: str, operation: str) -> Optional[Any]:
        """Try API call with fallback to other keys on failure"""
        # If we've previously determined all keys are exhausted and set a next_retry_time,
        # avoid attempting calls until that ETA has passed. This prevents aggressive
        # rotation and repeated owner notifications.
        if self.next_retry_time and datetime.now() < self.next_retry_time:
            logger.warning(f"Skipping API call for {operation}: keys exhausted until {self.next_retry_time}")
            return None

        start_index = self.current_model_index
        tried_indices = set()

        while len(tried_indices) < len(self.models):
            if not self.models:
                return None

            current_model_data = self.models[self.current_model_index]
            model = current_model_data['model']
            
            try:
                response = await asyncio.to_thread(model.generate_content, prompt)
                
                # Reset error count on success
                current_model_data['error_count'] = 0
                current_model_data['last_success'] = datetime.now()
                current_model_data['last_error'] = None
                
                return response
                
            except Exception as e:
                error_msg = str(e)
                current_model_data['last_error'] = error_msg
                current_model_data['error_count'] += 1
                
                # Special handling for rate limit errors
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(f"API key {self.current_model_index + 1} rate limited, rotating to next key")
                    if current_model_data['error_count'] >= 3:  # After 3 rate limit errors, notify owner
                        if config.owner_whatsapp_number:
                            try:
                                async with EnhancedWhatsAppClient() as client:
                                    await client.send_message(
                                        config.owner_whatsapp_number,
                                        f"⚠️ Gemini API key {self.current_model_index + 1} has been rate limited multiple times."
                                    )
                            except Exception as notify_err:
                                logger.error(f"Failed to notify owner about rate limit: {notify_err}")
                
                tried_indices.add(self.current_model_index)
                # Rotate to next key
                self.current_model_index = (self.current_model_index + 1) % len(self.models)
                
                # If we've tried all keys, check if it's been long enough since last try of first key
                if len(tried_indices) == len(self.models):
                    # If all keys are rate limited, wait a bit before allowing retry of first key
                    if datetime.now() - self.last_key_rotation > timedelta(minutes=5):
                        self.last_key_rotation = datetime.now()
                        tried_indices.clear()  # Allow trying keys again
                        continue
                    else:
                        logger.error(f"{operation} failed - all API keys exhausted: {e}")
                        # Notify owner that all keys are currently exhausted and set an ETA
                        self.next_retry_time = datetime.now() + timedelta(minutes=5)

                        # Mark AI as temporarily unavailable so other code paths use fallbacks
                        self.ai_available = False

                        # Only notify the owner once per exhaustion event
                        if config.owner_whatsapp_number and not self.owner_notified:
                            self.owner_notified = True
                            # Build a concise, sanitized status for the owner (no raw error dumps)
                            # Only include per-key error counts to avoid leaking verbose provider messages.
                            err_lines = [f"Key {idx+1}: errors={m.get('error_count', 0)}" for idx, m in enumerate(self.models)]
                            # Join into a single, short status string separated by bullets
                            status_msg = "  ".join(err_lines)
                            # Truncate aggressively if it's unexpectedly long
                            if len(status_msg) > 3000:
                                status_msg = status_msg[:2996] + "...[truncated]"

                            try:
                                eta_str = self.next_retry_time.strftime('%Y-%m-%d %H:%M:%S UTC') if self.next_retry_time else 'soon'
                                short_msg = (
                                    f"⚠️ Gemini API keys exhausted. Next retry ETA: {eta_str}.\n"
                                    f"Status: {status_msg}"
                                )
                                async with EnhancedWhatsAppClient() as client:
                                    await client.send_message(config.owner_whatsapp_number, short_msg)
                            except Exception as notify_err:
                                logger.error(f"Failed to notify owner about all-keys exhaustion: {notify_err}")

                        # Schedule a background task to re-enable AI after the ETA
                        try:
                            if self._reenable_task is None or self._reenable_task.done():
                                self._reenable_task = asyncio.create_task(self._reenable_after_eta())
                        except Exception as task_err:
                            logger.error(f"Failed to schedule re-enable task: {task_err}")

                        # Try OpenRouter fallback here before giving up entirely
                        try:
                            gen = await self._attempt_openrouter_fallback(prompt, operation)
                            if gen:
                                return gen
                        except Exception as e:
                            logger.error(f"OpenRouter fallback attempt crashed: {e}")

                        # Break out of rotation loop since all keys are exhausted
                        break

        return None

    async def analyze_message(self, message: str, context: Dict) -> MessageAnalysis:
        """Advanced message analysis using AI"""
        if not self.ai_available or not self.models:
            return self._fallback_analysis(message)
        
        analysis_prompt = f"""
Analyze this WhatsApp message comprehensively:

Message: "{message}"
Context: {json.dumps(context, default=str)}

Provide analysis as JSON:
{{
    "sentiment"7: float (-1 to 1),
    "emotions": {{"joy": 0.0, "anger": 0.0, "fear": 0.0, "sadness": 0.0, "surprise": 0.0}},
    "intent": "string",
    "urgency": float (0 to 1),
    "business_relevance": float (0 to 1),
    "language": "string",
    "contains_pii": boolean,
    "spam_score": float (0 to 1)
}}
        """
        
        try:
            response = await self._try_api_call(analysis_prompt, "Message analysis")
            if response and hasattr(response, 'text'):
                result = self._parse_json_response(response.text)
                if result:
                    return MessageAnalysis(**result)
            
        except Exception as e:
            logger.error(f"Message analysis failed: {e}")
        
        # Fallback analysis
        return self._fallback_analysis(message)
    
    def _fallback_analysis(self, message: str) -> MessageAnalysis:
        """Fallback analysis when AI is not available"""
        # Simple keyword-based analysis
        message_lower = message.lower()
        
        # Basic sentiment detection
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'awesome', 'perfect']
        negative_words = ['bad', 'terrible', 'hate', 'angry', 'frustrated', 'disappointed']
        
        positive_score = sum(1 for word in positive_words if word in message_lower)
        negative_score = sum(1 for word in negative_words if word in message_lower)
        
        if positive_score > negative_score:
            sentiment = 0.5
        elif negative_score > positive_score:
            sentiment = -0.5
        else:
            sentiment = 0.0
        
        # Basic urgency detection
        urgent_words = ['urgent', 'asap', 'emergency', 'immediately', 'now', 'quick']
        urgency = 0.8 if any(word in message_lower for word in urgent_words) else 0.3
        
        # Basic business relevance
        business_words = ['business', 'deal', 'investment', 'money', 'project', 'work', 'meeting']
        business_relevance = 0.7 if any(word in message_lower for word in business_words) else 0.4
        
        # Basic spam detection
        spam_indicators = ['win', 'lottery', 'prize', 'click here', 'free money', 'congratulations']
        spam_score = 0.8 if any(indicator in message_lower for indicator in spam_indicators) else 0.1
        
        return MessageAnalysis(
            sentiment=sentiment,
            emotions={"joy": max(0, sentiment), "anger": max(0, -sentiment)},
            intent="general_inquiry",
            urgency=urgency,
            business_relevance=business_relevance,
            language="en",
            contains_pii=bool(re.search(r'\b\d{10,}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)),
            spam_score=spam_score
        )
    
    async def classify_priority_and_category(self, message: str, contact: ContactInfo, analysis: MessageAnalysis) -> Tuple[Priority, Category]:
        """Advanced priority and category classification"""
        
        # VIP contacts get higher priority
        base_priority = Priority.HIGH if contact.is_vip else Priority.MEDIUM
        
        # Adjust based on analysis
        if analysis.urgency > 0.8 or analysis.business_relevance > 0.8:
            priority = Priority.HIGH
        elif analysis.spam_score > 0.7:
            priority = Priority.SPAM
        elif analysis.urgency < 0.2 and analysis.business_relevance < 0.3:
            priority = Priority.LOW
        else:
            priority = base_priority
        
        # Determine category based on intent and content
        if not self.ai_available:
            category = self._fallback_categorization(message, analysis)
        else:
            category_prompt = f"""
Categorize this business message:
Message: "{message}"
Intent: {analysis.intent}
Business Relevance: {analysis.business_relevance}

Choose ONE category: {[cat.value for cat in Category]}
Respond with only the category name.
            """
            
            try:
                # Use rotation-aware API call
                if self.ai_available and self.models:
                    response = await self._try_api_call(category_prompt, "Category classification")
                    category_text = response.text.strip().lower() if response and hasattr(response, 'text') else ''

                    for cat in Category:
                        if cat.value.lower() in category_text:
                            category = cat
                            break
                    else:
                        category = Category.GENERAL
                else:
                    category = Category.GENERAL
                
            except Exception as e:
                logger.error(f"Category classification failed: {e}")
                category = self._fallback_categorization(message, analysis)
        
        return priority, category
    
    def _fallback_categorization(self, message: str, analysis: MessageAnalysis) -> Category:
        """Fallback categorization when AI is not available"""
        message_lower = message.lower()
        
        if analysis.spam_score > 0.7:
            return Category.SPAM
        elif any(word in message_lower for word in ['investment', 'money', 'fund', 'capital']):
            return Category.INVESTMENT
        elif any(word in message_lower for word in ['business', 'deal', 'partnership']):
            return Category.BUSINESS
        elif any(word in message_lower for word in ['help', 'support', 'problem', 'issue']):
            return Category.SUPPORT
        elif any(word in message_lower for word in ['network', 'connect', 'introduce']):
            return Category.NETWORKING
        elif any(word in message_lower for word in ['buy', 'sell', 'product', 'service']):
            return Category.SALES
        else:
            return Category.GENERAL
    
    async def generate_response(self, message: str, contact: ContactInfo, conversation_history: List[Dict], analysis: MessageAnalysis) -> str:
        """Generate intelligent AI response"""
        
        # If contact is unknown, immediately ask for name in the detected language
        lang = getattr(analysis, 'language', 'en') if analysis is not None else 'en'
        if not contact.name:
            return self._get_quick_response("unknown_contact", lang=lang)
        
        if not self.ai_available or not self.models:
            return self._get_fallback_response(contact, analysis)
        
        # Build context
        context = self._build_conversation_context(conversation_history, contact, analysis)
        
        # Generate response using AI
        response_prompt = f"""
{self.system_prompt}

CURRENT CONTEXT:
- Contact: {contact.name or 'Unknown'} ({contact.phone_number})
- Known Contact: {contact.is_known}
- VIP: {contact.is_vip}
- Previous Interactions: {contact.interaction_count}
- Message Sentiment: {analysis.sentiment:.2f}
- Intent: {analysis.intent}
- Urgency: {analysis.urgency:.2f}
- Business Relevance: {analysis.business_relevance:.2f}

CONVERSATION HISTORY:
{context}

NEW MESSAGE: "{message}"

Generate a response that:
1. Matches the urgency and importance of the message
2. Maintains professional boundaries
3. Provides clear next steps if applicable
4. Stays concise but complete
5. Reflects the personality described above

RESPONSE:
        """
        
        try:
            # Add explicit style directives (language, formality, verbosity) if present
            style_directive = ''
            try:
                style = getattr(analysis, 'style', None) or {}
                lang = style.get('language') if isinstance(style, dict) else getattr(analysis, 'language', 'en')
                formality = style.get('formality', 'informal') if isinstance(style, dict) else 'informal'
                verbosity = style.get('verbosity', 'normal') if isinstance(style, dict) else 'normal'

                # Language directive
                if lang == 'sw':
                    style_directive += "\n\nPlease respond in fluent, natural Swahili (Kiswahili)."
                else:
                    style_directive += "\n\nPlease respond in clear English."

                # Formality directive
                if formality == 'formal':
                    style_directive += " Use a moderately formal tone."
                else:
                    style_directive += " Use a friendly, conversational tone."

                # Verbosity directive
                if verbosity == 'short':
                    style_directive += " Keep the reply very short (1-2 short sentences)."
                elif verbosity == 'detailed':
                    style_directive += " Provide details concisely (2-4 sentences)."
                else:
                    style_directive += " Keep the reply concise and to the point."
            except Exception:
                style_directive = "\n\nPlease respond in English. Keep the reply concise."

            response = await self._try_api_call(response_prompt + style_directive, "Response generation")
            if response and hasattr(response, 'text'):
                generated_response = response.text.strip()
                # Post-process response
                return self._post_process_response(generated_response, contact, analysis)
            
            logger.error("AI response missing text attribute")
            return self._get_fallback_response(contact, analysis)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._get_fallback_response(contact, analysis)
    
    def _build_conversation_context(self, history: List[Dict], contact: ContactInfo, analysis: MessageAnalysis) -> str:
        """Build conversation context for AI"""
        if not history:
            return "No previous messages"
        
        # Get last 5 messages for context
        recent_messages = history[-5:]
        context_lines = []
        
        for msg in recent_messages:
            # Ensure correct precedence: if message is from user, show contact name (or 'Contact'), else 'Assistant'
            if msg.get('is_from_user'):
                sender = contact.name or "Contact"
            else:
                sender = "Assistant"
            content = msg.get('content', '')[:100]  # Truncate long messages
            timestamp = msg.get('timestamp', '')
            if isinstance(timestamp, str) and 'T' in timestamp:
                timestamp = timestamp.split('T')[0]
            elif hasattr(timestamp, 'date'):
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.date().isoformat()
            else:
                timestamp = 'Unknown'
            context_lines.append(f"{sender} ({timestamp}): {content}")
        
        return "\n".join(context_lines)
    
    def _get_quick_response(self, response_type: str, lang: str = 'en') -> str:
        """Get a random quick response of specified type and language (en|sw)"""
        import random
        bucket = self.quick_responses.get(response_type, self.quick_responses["fallback"])
        # Choose language-specific list if present
        if isinstance(bucket, dict):
            choices = bucket.get(lang, bucket.get('en', []))
        else:
            choices = bucket
        if not choices:
            choices = self.quick_responses['fallback'].get('en', []) if isinstance(self.quick_responses['fallback'], dict) else self.quick_responses['fallback']
        return random.choice(choices) if choices else ""

    def _detect_language(self, text: str) -> str:
        """Very small heuristic language detector: returns 'sw' for Swahili-like text, else 'en'.
        This is intentionally simple and covers common greetings/words to force the model to reply in Swahili.
        """
        if not text:
            return 'en'
        s = text.strip().lower()
        # common swahili words/greetings
        swahili_indicators = ['habari', 'hujambo', 'mambo', 'asante', 'sawa', 'karibu', 'ndio', 'hapana', 'tafadhali', 'salama']
        for w in swahili_indicators:
            if w in s:
                return 'sw'
        # fallback: detect presence of accented characters or common english words
        return 'en'
    
    def _post_process_response(self, response: str, contact: ContactInfo, analysis: MessageAnalysis) -> str:
        """Post-process AI response for quality and safety"""
        
        # Remove any potential personal information
        response = re.sub(r'\b\d{10,}\b', '[PHONE_REDACTED]', response)
        response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', response)
        
        # Ensure response isn't too long
        if len(response) > 500:
            response = response[:500] + "..."
        
        # Add business hours context if outside hours
        current_hour = datetime.now().hour
        if current_hour < config.business_start_hour or current_hour > config.business_end_hour:
            if analysis.urgency < 0.7:  # Not urgent
                response += f"\n\nNote: I typically respond during business hours ({config.business_start_hour}:00-{config.business_end_hour}:00). For urgent matters, please indicate the urgency level."
        
        return response.strip()

    async def _validate_and_fix_swahili(self, text: str, analysis: MessageAnalysis, contact: ContactInfo, conversation_history: List[Dict]) -> str:
        """Validate a Swahili response for basic fluency and retry generation if it looks wrong.

        Heuristic: count presence of common Swahili words. If below threshold, attempt one regeneration
        by asking the model to produce a short, natural Swahili reply without invented words.
        """
        try:
            lang = getattr(analysis, 'language', 'en')
            if lang != 'sw':
                return text

            # Quick heuristic: token overlap with common words
            words = re.findall(r"[\w']+", text.lower())
            if not words:
                return text

            common_hits = sum(1 for w in words if w in self.swahili_common_words)
            ratio = common_hits / max(1, len(words))

            # If ratio is acceptable (>= 0.12), assume it's reasonably Swahili
            if ratio >= 0.12:
                return text

            # Otherwise, attempt one short regeneration pass with a strict instruction
            regen_prompt = (
                f"The earlier reply may contain spelling mistakes or invented words in Kiswahili. "
                f"Please rewrite the reply in clear, natural Kiswahili. Use simple correct words and avoid inventing new terms. "
                f"Be concise (1-3 short sentences). Original reply: {text}"
            )

            # Use rotation-aware API call; if AI not available, return original text
            if not self.ai_available or not self.models:
                return text

            try:
                response = await self._try_api_call(regen_prompt, "Swahili regeneration")
                if response and hasattr(response, 'text'):
                    new_text = response.text.strip()
                    # If new_text appears better by heuristic, use it
                    words2 = re.findall(r"[\w']+", new_text.lower())
                    common_hits2 = sum(1 for w in words2 if w in self.swahili_common_words)
                    if common_hits2 / max(1, len(words2)) >= 0.12:
                        return new_text
            except Exception as e:
                logger.debug(f"Swahili regeneration failed: {e}")

            return text
        except Exception as e:
            logger.error(f"_validate_and_fix_swahili error: {e}")
            return text
    
    def _get_fallback_response(self, contact: ContactInfo, analysis: MessageAnalysis) -> str:
        """Fallback response when AI generation fails"""
        # Determine language
        lang = getattr(analysis, 'language', 'en') if analysis is not None else 'en'

        # If we've set an ETA for retrying API keys, return an agent-unavailable message
        if hasattr(self, 'next_retry_time') and self.next_retry_time:
            eta = self.next_retry_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            # Warm, calm phrasing regardless of whether contact name is known
            if lang == 'sw':
                if contact and getattr(contact, 'name', None):
                    return f"Habari {contact.name}, kwa sasa siko tayari kutokana na mahitaji mengi. Tafadhali jaribu tena baadaye. Asante kwa uvumilivu wako. (ETA: {eta})"
                else:
                    return f"Habari — kwa sasa siko tayari kutokana na mahitaji mengi. Tafadhali jaribu tena baadaye. Asante kwa uvumilivu. (ETA: {eta})"
            else:
                if contact and getattr(contact, 'name', None):
                    return f"Hello {contact.name}, I'm currently temporarily unavailable due to high demand. Please try again later 🙂. Thank you for your patience. (ETA: {eta})"
                else:
                    return f"Hi there — I'm currently temporarily unavailable due to high demand. Please try again later 🙂. Thank you for your patience. (ETA: {eta})"

        if not contact.name:
            return "Asante kwa ujumbe wako. Tafadhali nitaje jina lako na niambie niwezaje kukusaidia." if lang == 'sw' else "Thank you for your message. Could you please tell me your name and how I can assist you?"

        if analysis and getattr(analysis, 'urgency', 0) > 0.7:
            return f"Habari {contact.name}, ninaelewa hili linaweza kuwa la dharura. Nitakagua ujumbe wako na nitajibu hivi karibuni." if lang == 'sw' else f"Hello {contact.name}, I understand this may be urgent. I'll review your message and respond shortly."
        
        return f"Habari {contact.name}, asante kwa ujumbe wako. Nitarudi kwako hivi karibuni." if lang == 'sw' else f"Hello {contact.name}, thank you for your message. I'll get back to you with a proper response soon."
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Parse JSON response from AI, handling various formats"""
        try:
            return json.loads(text)
        except:
            try:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        return None
    
    async def generate_conversation_summary(self, messages: List[Dict], contact: ContactInfo) -> Optional[ConversationSummary]:
        """Generate comprehensive conversation summary"""
        
        if not messages or not self.ai_available or not self.models:
            return None
        
        # Prepare message text
        conversation_text = "\n".join([
            f"{'Contact' if msg.get('is_from_user') else 'Assistant'}: {msg.get('content', '')}"
            for msg in messages[-20:]  # Limit to last 20 messages to avoid token limits
        ])
        
        summary_prompt = f"""
Analyze this complete WhatsApp conversation and provide comprehensive summary:

Contact: {contact.name or 'Unknown'} ({contact.phone_number})
Messages: {len(messages)}

Conversation:
{conversation_text}

Provide summary as JSON:
{{
    "summary": "Brief overall summary",
    "key_topics": ["topic1", "topic2", "topic3"],
    "action_items": ["action1", "action2"],
    "sentiment_trend": [0.2, 0.5, 0.8],
    "satisfaction_score": 0.85,
    "resolution_status": "resolved|pending|escalated"
}}
        """
        
        try:
            response = await self._try_api_call(summary_prompt, "Summary generation")
            if not response or not hasattr(response, 'text'):
                logger.error("AI response missing text attribute")
                return None
                
            result = self._parse_json_response(response.text)
            
            if result:
                start_time = messages[0].get('timestamp')
                end_time = messages[-1].get('timestamp')
                
                # Handle timestamp parsing
                try:
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    elif not isinstance(start_time, datetime):
                        start_time = datetime.now(timezone.utc)
                    
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    elif not isinstance(end_time, datetime):
                        end_time = datetime.now(timezone.utc)
                except Exception as e:
                    logger.error(f"Error parsing timestamps: {e}")
                    start_time = datetime.now(timezone.utc)
                    end_time = datetime.now(timezone.utc)
                
                duration = int((end_time - start_time).total_seconds() / 60)
                
                return ConversationSummary(
                    conversation_id=messages[0].get('conversation_id', str(uuid.uuid4())),
                    contact_info=contact,
                    message_count=len(messages),
                    priority=Priority.MEDIUM,  # Could be determined from analysis
                    category=Category.GENERAL,  # Could be determined from analysis
                    summary=result.get('summary', ''),
                    key_topics=result.get('key_topics', []),
                    action_items=result.get('action_items', []),
                    sentiment_trend=result.get('sentiment_trend', []),
                    start_time=start_time,
                    end_time=end_time,
                    duration_minutes=duration,
                    satisfaction_score=result.get('satisfaction_score', 0.0),
                    resolution_status=result.get('resolution_status', 'pending')
                )
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
        
        return None

    async def is_conversation_ending(self, recent_messages: List[Dict], contact: ContactInfo) -> Tuple[bool, Optional[str]]:
        """Decide whether the conversation is ending using the AI model when available.

        Returns (is_ending, reason). Falls back to simple heuristics when AI is unavailable or fails.
        """
        # Basic fallback heuristics
        def heuristics(msgs: List[Dict]) -> Tuple[bool, Optional[str]]:
            if not msgs:
                return False, None
            last = msgs[-1].get('content', '')
            if not isinstance(last, str):
                last = str(last)
            last_lower = last.lower()
            goodbye_keywords = ['bye', 'goodbye', 'see you', 'see ya', 'have a nice', 'have a good', 'thanks', 'thank you', 'ok', 'ok bye', 'got it']
            if any(k in last_lower for k in goodbye_keywords):
                return True, 'heuristic_keyword'

            # Short affirmative replies after a meeting confirmation are likely closures
            if len(last_lower.split()) <= 3 and any(w in last_lower for w in ['ok', 'okay', 'sure', 'thanks', 'thank']):
                return True, 'heuristic_short_ack'

            return False, None

        # If AI not available, use heuristics
        if not self.ai_available or not self.models:
            return heuristics(recent_messages)

        # Build a compact prompt for the model to decide if the conversation is ending
        try:
            # Convert datetime objects to ISO format strings in messages
            compact_msgs = []
            for m in (recent_messages[-8:] if recent_messages else []):
                msg_copy = {}
                for k, v in m.items():
                    if isinstance(v, datetime):
                        msg_copy[k] = v.isoformat()
                    else:
                        msg_copy[k] = v
                compact_msgs.append({
                    "sender": (msg_copy.get('is_from_user') and 'user' or 'assistant'),
                    "text": msg_copy.get('content', '')
                })

            # Convert contact object to serializable dict
            contact_dict = {}
            if hasattr(contact, 'dict'):
                contact_dict = contact.dict()
                # Convert any datetime objects in contact dict
                for k, v in contact_dict.items():
                    if isinstance(v, datetime):
                        contact_dict[k] = v.isoformat()

            prompt = f"""
You are a helpful assistant that decides whether a short WhatsApp conversation has been ended by the user.
Return only a JSON object with two fields: {{"ending": true|false, "reason": "short explanation"}}.

Messages: {json.dumps(compact_msgs, ensure_ascii=False)}
Contact: {json.dumps(contact_dict, ensure_ascii=False)}

Consider the conversation is ending when the user signals goodbye, confirms a meeting then says bye/ok, or otherwise clearly closes the interaction. Do not invent new facts.
            """

            response = await self._try_api_call(prompt, "Conversation ending check")
            text = response.text if response and hasattr(response, 'text') else str(response)
            parsed = self._parse_json_response(text)
            if parsed and isinstance(parsed, dict) and 'ending' in parsed:
                return bool(parsed.get('ending')), parsed.get('reason')

            # Simple textual fallback from model output
            text_lower = text.lower()
            if 'yes' in text_lower or 'true' in text_lower:
                return True, 'model_hint'
            if 'no' in text_lower or 'false' in text_lower:
                return False, 'model_hint'

        except Exception as e:
            logger.error(f"is_conversation_ending model check failed: {e}")

        # Final fallback
        return heuristics(recent_messages)

    async def _reenable_after_eta(self):
        """Background task that waits until next_retry_time and re-enables AI."""
        try:
            if not self.next_retry_time:
                return

            now = datetime.now()
            wait_seconds = (self.next_retry_time - now).total_seconds()
            if wait_seconds > 0:
                logger.info(f"AI re-enable scheduled in {int(wait_seconds)} seconds")
                await asyncio.sleep(wait_seconds)

            # Try to reinitialize models (in case some keys recovered)
            try:
                self.init_ai()
            except Exception as e:
                logger.error(f"Error reinitializing AI after ETA: {e}")

            # Reset flags
            self.next_retry_time = None
            self.owner_notified = False
            self.ai_available = bool(self.models)
            logger.info("AI re-enabled after ETA; models available: %s", len(self.models))
        except Exception as e:
            logger.error(f"_reenable_after_eta failed: {e}")

    async def _attempt_openrouter_fallback(self, prompt: str, operation: str):
        """Attempt generation via OpenRouter as a fallback when Gemini is exhausted."""
        if not config.enable_openrouter_fallback:
            logger.info("OpenRouter fallback disabled by configuration")
            return None

        try:
            logger.info(f"Attempting OpenRouter fallback for operation: {operation}; openrouter_model={config.openrouter_model}")
            try:
                or_client = OpenRouterClient(model=config.openrouter_model)
            except Exception as e:
                logger.warning(f"OpenRouter client init failed: {e}")
                return None

            # Log number of keys available for rotation if possible
            try:
                keys_count = len(getattr(or_client, 'keys', []))
                logger.info(f"OpenRouter client initialized with {keys_count} keys")
            except Exception:
                pass

            timeout_val = getattr(config, 'openrouter_timeout', 20)
            gen = await or_client.generate(prompt, timeout=timeout_val)
            if gen and getattr(gen, 'text', None):
                logger.info("OpenRouter fallback returned a response")
                try:
                    setattr(gen, 'provider', 'openrouter')
                except Exception:
                    pass
                return gen
        except Exception as e:
            logger.warning(f"OpenRouter fallback error: {e}")

        return None

# Initialize AI handler
ai_handler = AdvancedAIHandler()

# Enhanced WhatsApp Client
class EnhancedWhatsAppClient:
    def __init__(self):
        self.base_url = "https://graph.facebook.com/v18.0"
        self.access_token = config.whatsapp_access_token
        self.phone_number_id = config.whatsapp_phone_number_id
        self.http_client = None
    
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()
    
    async def send_message(self, to: str, message: str, message_type: str = "text") -> Dict:
        """Enhanced message sending with better error handling"""
        
        # Sanitize phone number
        phone_digits = ''.join(c for c in str(to) if c.isdigit())
        if not phone_digits:
            raise HTTPException(400, "Invalid phone number format")
        
        # DEV / TEST short-circuit
        if config.disable_whatsapp_sends:
            logger.info("DISABLE_WHATSAPP_SENDS enabled - returning mock response without network call")
            return {
                "messages": [{"id": f"mock_{uuid.uuid4()}"}],
                "meta": {"api_status": "disabled_by_config"}
            }

        if not self.access_token or not self.phone_number_id:
            logger.error("WhatsApp API not configured in production mode")
            raise HTTPException(500, "WhatsApp API not configured")
        
        # Validate recipient against allowlist if configured
        allowed = config.whatsapp_allowed_recipients
        if allowed and phone_digits not in allowed:
            logger.error(f"Recipient {phone_digits} not in allowed recipients list")
            raise HTTPException(403, "Recipient phone number not allowed")

        url = f"{self.base_url}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": phone_digits,
            "type": message_type,
        }
        
        if message_type == "text":
            payload["text"] = {"body": message}
        
        # Perform retries with exponential backoff for transient errors
        max_retries = max(1, config.whatsapp_send_max_retries)
        base_backoff = float(config.whatsapp_send_backoff_base or 0.5)

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                client = self.http_client or httpx.AsyncClient(timeout=30.0)
                # Use client as context manager if we created it here
                if self.http_client is None:
                    async with client as c:
                        response = await c.post(url, headers=headers, json=payload)
                else:
                    response = await client.post(url, headers=headers, json=payload)

                # Raise for HTTP errors
                response.raise_for_status()
                result = response.json()

                logger.info(f"Message sent successfully to {phone_digits[:4]}**** (attempt {attempt})")
                return result

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text
                # Permanent client errors (4xx) should not be retried except 429
                if status in (400, 401, 403, 404):
                    logger.error(f"Permanent WhatsApp API error {status}: {body}")
                    raise HTTPException(500, f"WhatsApp API error {status}: {body}")

                # Rate limit or server errors: retry
                last_exc = e
                logger.warning(f"Transient WhatsApp API error {status} on attempt {attempt}: {body}")

            except Exception as e:
                last_exc = e
                logger.warning(f"WhatsApp send attempt {attempt} failed: {e}")

            # Backoff before next attempt
            if attempt < max_retries:
                backoff = base_backoff * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

        # If we reach here, all retries failed
        logger.error(f"All WhatsApp send attempts failed for {phone_digits}")
        raise HTTPException(500, f"Failed to send WhatsApp message after {max_retries} attempts: {last_exc}")
    
    async def get_media(self, media_id: str) -> Dict:
        """Download media from WhatsApp"""
        if not self.access_token:
            logger.warning("WhatsApp API not configured")
            return {"error": "API not configured"}
        
        url = f"{self.base_url}/{media_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            if not self.http_client:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=headers)
            else:
                response = await self.http_client.get(url, headers=headers)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Media download failed: {e}")
            raise HTTPException(500, f"Failed to download media: {str(e)}")
        

# Business Logic Managers
class ContactManager:
    """Advanced contact management with learning capabilities"""

    async def get_or_create_contact(self, phone_number: str) -> ContactInfo:
        """Get existing contact or create new one via services.persistence"""
        try:
            from services.persistence import get_or_create_contact as _get_contact
            doc = await _get_contact(phone_number, db_manager, cache_manager)
            # Convert dict to ContactInfo if possible
            try:
                return ContactInfo(**doc)
            except Exception:
                # Fallback minimal ContactInfo
                return ContactInfo(phone_number=phone_number, priority_level=Priority.MEDIUM, created_at=datetime.now(timezone.utc))
        except Exception as e:
            logger.error(f"get_or_create_contact delegation failed: {e}")
            return ContactInfo(phone_number=phone_number, priority_level=Priority.MEDIUM, created_at=datetime.now(timezone.utc))

    async def update_contact(self, phone_number: str, updates: Dict) -> ContactInfo:
        """Update contact via services.persistence and return ContactInfo"""
        try:
            from services.persistence import update_contact as _update_contact
            updated = await _update_contact(phone_number, updates, db_manager, cache_manager)
            if updated:
                try:
                    return ContactInfo(**updated)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"update_contact delegation failed: {e}")

        # Fallback: return existing contact object
        contact = await self.get_or_create_contact(phone_number)
        for key, value in updates.items():
            if hasattr(contact, key):
                setattr(contact, key, value)
        return contact

class ConversationManager:
    """Advanced conversation management with context awareness"""
    
    async def get_or_create_conversation(self, phone_number: str) -> str:
        """Get active conversation or create new one"""
        try:
            from services.persistence import get_or_create_conversation as _get_conv
            conv_id = await _get_conv(phone_number, db_manager, cache_manager)
            return conv_id
        except Exception as e:
            logger.error(f"get_or_create_conversation delegation failed: {e}")
            # Fallback: create a simple uuid
            conv_id = str(uuid.uuid4())
            return conv_id
    
    async def get_conversation_context(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history with intelligent context"""
        try:
            from services.persistence import get_conversation_context as _get_context
            msgs = await _get_context(conversation_id, limit, db_manager)
            return msgs
        except Exception as e:
            logger.error(f"get_conversation_context delegation failed: {e}")
            return []

class MessageProcessor:
    """Advanced message processing pipeline"""
    
    def __init__(self):
        self.contact_manager = ContactManager()
        self.conversation_manager = ConversationManager()
    
    async def process_message(self, webhook_data: Dict) -> bool:
        """Main message processing pipeline"""
        
        start_time = datetime.now()
        
        try:
            # Extract message data
            message_data = self._extract_message_data(webhook_data)
            if not message_data:
                return False
            
            # Check for duplicates and rate limits
            if not self._validate_message(message_data):
                return False
            
            # Get or create contact
            contact = await self.contact_manager.get_or_create_contact(
                message_data['phone_number']
            )
            
            # Get conversation
            conversation_id = await self.conversation_manager.get_or_create_conversation(
                message_data['phone_number']
            )
            
            # Detect language early using the AI handler's heuristic and include it in analysis context
            try:
                detected_lang = ai_handler._detect_language(message_data.get('content', ''))
            except Exception:
                detected_lang = 'en'

            # Quick pass: detect self-introductions like 'Naitwa <name>' and immediate meeting intent.
            try:
                intro_handled = False
                intro_name = None
                intro_text = message_data.get('content', '') or ''
                # look for 'naitwa <name>' or 'jina ni <name>' patterns (case-insensitive)
                m = re.search(r"\b(?:naitwa|jina\s+ni)\s+([A-Za-z\u00C0-\u017F]+)", intro_text, flags=re.IGNORECASE)
                if m:
                    intro_name = m.group(1).strip()
                    # Save name on contact right away
                    try:
                        await self.contact_manager.update_contact(message_data['phone_number'], {"name": intro_name, "is_known": True})
                        # refresh contact object
                        contact = await self.contact_manager.get_or_create_contact(message_data['phone_number'])
                    except Exception as e:
                        logger.debug(f"Failed to persist intro name: {e}")

                    # Check if message expresses wanting to meet (nataka/kuona/kuonana)
                    if re.search(r"\b(nataka|taka|kuona|kuonana)\b", intro_text, flags=re.IGNORECASE):
                        # craft a concise Swahili acknowledgement/request for scheduling details
                        try:
                            sw_reply = f"Habari {intro_name}, asante. Ninaona ungependa kumwona 'mkubwa' — unaweza nifafanulie ni nani unataka kumwona na ni lini ungependelea?"
                            async with EnhancedWhatsAppClient() as client:
                                await client.send_message(message_data['phone_number'], sw_reply)
                            # save outgoing reply
                            response_message = EnhancedMessage(
                                message_id=str(uuid.uuid4()),
                                conversation_id=conversation_id,
                                phone_number=message_data['phone_number'],
                                message_type=MessageType.TEXT,
                                content=sw_reply,
                                timestamp=datetime.now(timezone.utc),
                                is_from_user=False,
                                priority=Priority.MEDIUM,
                                category=Category.GENERAL
                            )
                            await self._save_message(response_message)
                            logger.info(f"Handled self-introduction and meeting intent for {message_data['phone_number']}")
                            return True
                        except Exception as e:
                            logger.error(f"Failed to send self-intro acknowledgement: {e}")
                            # fall through to normal processing if send fails
                # end intro detection
            except Exception:
                pass

            # Analyze message (include detected language in context)
            analysis = await ai_handler.analyze_message(
                message_data['content'],
                {"contact": contact.dict(), "detected_language": detected_lang}
            )

            # Enforce our detection if model didn't return a clear language
            try:
                if not getattr(analysis, 'language', None) or (detected_lang == 'sw' and analysis.language != 'sw'):
                    analysis.language = detected_lang
            except Exception:
                pass

            # Resolve style preferences (language/formality/verbosity) per-contact
            try:
                style = await self._resolve_style(contact, analysis, message_data['content'])
                # allow downstream code to access style via analysis object
                try:
                    analysis.style = style
                except Exception:
                    pass
            except Exception:
                style = {}

            # If AI exhaustion was detected during analysis, first try OpenRouter fallback
            # rotating over available OpenRouter keys. If that fails, send the existing ETA fallback.
            if getattr(ai_handler, 'next_retry_time', None) and not ai_handler.ai_available:
                openrouter_response_text = None
                try:
                    # Only attempt OpenRouter if enabled in config
                    if not config.enable_openrouter_fallback:
                        or_client = None
                    else:
                        # Build a concise fallback prompt that preserves language and style hints
                        lang_hint = getattr(analysis, 'language', 'en')
                        style = getattr(analysis, 'style', {}) or {}
                        formality = style.get('formality', 'friendly')
                        verbosity = style.get('verbosity', 'concise')

                        style_directives = []
                        if lang_hint == 'sw':
                            style_directives.append('Respond in fluent, natural Kiswahili (Swahili).')
                        else:
                            style_directives.append('Respond in clear English.')

                        if formality == 'formal':
                            style_directives.append('Use a moderately formal tone.')
                        else:
                            style_directives.append('Use a friendly, conversational tone.')

                        if verbosity == 'short' or verbosity == 'concise':
                            style_directives.append('Keep the reply short (1-3 sentences).')
                        elif verbosity == 'detailed':
                            style_directives.append('Provide a detailed but concise reply (3-5 sentences).')

                        fallback_prompt = (
                            f"{ai_handler.system_prompt}\n\nCONTACT: {contact.name or contact.phone_number} ({contact.phone_number})\n"
                            f"USER_MESSAGE: {message_data['content']}\n\n"
                            + " ".join(style_directives)
                        )

                        # Initialize OpenRouter client (it will auto-rotate keys)
                        try:
                            or_client = OpenRouterClient(model=config.openrouter_model)
                        except Exception as e:
                            logger.warning(f"OpenRouter client unavailable: {e}")
                            or_client = None

                    if or_client:
                        try:
                            timeout_val = getattr(config, 'openrouter_timeout', 20)
                            gen = await or_client.generate(fallback_prompt, temperature=0.0, timeout=timeout_val)
                            if gen and getattr(gen, 'text', None):
                                openrouter_response_text = gen.text.strip()
                        except Exception as e:
                            logger.warning(f"OpenRouter fallback generation failed: {e}")

                except Exception as e:
                    logger.error(f"OpenRouter fallback attempt crashed: {e}")

                if openrouter_response_text:
                    # Post-process and send the OpenRouter response
                    try:
                        processed = ai_handler._post_process_response(openrouter_response_text, contact, analysis)
                        async with EnhancedWhatsAppClient() as client:
                            await client.send_message(message_data['phone_number'], processed)
                        logger.info(f"Sent OpenRouter fallback response to {message_data['phone_number']}")
                        # Save the response
                        response_message = EnhancedMessage(
                            message_id=str(uuid.uuid4()),
                            conversation_id=conversation_id,
                            phone_number=message_data['phone_number'],
                            message_type=MessageType.TEXT,
                            content=processed,
                            timestamp=datetime.now(timezone.utc),
                            is_from_user=False,
                            priority=Priority.MEDIUM,
                            category=Category.GENERAL
                        )
                        try:
                            response_message.provider_used = 'openrouter'
                            response_message.send_status = 'sent'
                        except Exception:
                            pass
                        await self._save_message(response_message)
                        return True
                    except Exception as e:
                        logger.error(f"Failed to send/save OpenRouter fallback response: {e}")

                # If OpenRouter also failed, send the original ETA-based fallback
                try:
                    fallback_text = ai_handler._get_fallback_response(contact, analysis)
                    async with EnhancedWhatsAppClient() as client:
                        await client.send_message(message_data['phone_number'], fallback_text)
                    logger.info(f"Sent fallback unavailable message to {message_data['phone_number']}")
                except Exception as e:
                    logger.error(f"Failed to send fallback unavailable message: {e}")

                # Save the fallback response as the agent's reply for record
                response_message = EnhancedMessage(
                    message_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    phone_number=message_data['phone_number'],
                    message_type=MessageType.TEXT,
                    content=fallback_text,
                    timestamp=datetime.now(timezone.utc),
                    is_from_user=False,
                    priority=Priority.MEDIUM,
                    category=Category.GENERAL
                )
                await self._save_message(response_message)

                # Update conversation last_activity minimally and stop further processing
                try:
                    from services.persistence import update_conversation_activity
                    await update_conversation_activity(conversation_id, db_manager, inc=1)
                except Exception as e:
                    logger.error(f"Error updating conversation after fallback (delegation failed): {e}")

                return True
            
            # Classify priority and category
            priority, category = await ai_handler.classify_priority_and_category(
                message_data['content'], contact, analysis
            )
            
            # Create enhanced message object
            enhanced_message = EnhancedMessage(
                message_id=message_data['message_id'],
                conversation_id=conversation_id,
                phone_number=message_data['phone_number'],
                message_type=MessageType(message_data.get('message_type', 'text')),
                content=message_data['content'],
                timestamp=datetime.now(timezone.utc),
                is_from_user=True,
                priority=priority,
                category=category,
                analysis=analysis,
                processing_time_ms=0  # Will be calculated
            )
            
            # Save message
            await self._save_message(enhanced_message)
            
            # Get conversation context
            conversation_history = await self.conversation_manager.get_conversation_context(
                conversation_id
            )
            
            # Generate AI response
            ai_response = await ai_handler.generate_response(
                message_data['content'],
                contact,
                conversation_history,
                analysis
            )
            
            # Send response
            try:
                async with EnhancedWhatsAppClient() as client:
                    await client.send_message(
                        message_data['phone_number'],
                        ai_response
                    )
            except Exception as e:
                logger.error(f"Failed to send WhatsApp message: {e}")
                # If recipient not allowed, mark message accordingly and notify owner
                msg_err = str(e)
                if 'Recipient phone number not in allowed list' in msg_err or 'not in allowed' in msg_err:
                    try:
                        # Save the outgoing message with failed send status
                        response_message.send_status = 'failed_recipient_not_allowed'
                        # Notify owner with concise info
                        if config.owner_whatsapp_number:
                            async with EnhancedWhatsAppClient() as owner_client:
                                await owner_client.send_message(
                                    config.owner_whatsapp_number,
                                    f"⚠️ Outgoing message to {message_data['phone_number']} blocked by WhatsApp recipient allowlist. Please add the number to your WhatsApp Business recipient list."
                                )
                    except Exception as notify_err:
                        logger.error(f"Failed to notify owner about blocked recipient: {notify_err}")
                # Continue processing even if send fails
            
            # Save AI response
            response_message = EnhancedMessage(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                phone_number=message_data['phone_number'],
                message_type=MessageType.TEXT,
                content=ai_response,
                timestamp=datetime.now(timezone.utc),
                is_from_user=False,
                priority=priority,
                category=category
            )
            
            await self._save_message(response_message)
            
            # Update conversation and contact
            await self._update_entities(conversation_id, contact, priority, category)
            
            # Use AI to decide if the conversation is ending. Fall back to heuristics inside the AI handler.
            try:
                conversation_history = await self.conversation_manager.get_conversation_context(conversation_id)
                is_ending, reason = await ai_handler.is_conversation_ending(conversation_history, contact)
            except Exception as e:
                logger.error(f"Error checking conversation end: {e}")
                is_ending, reason = False, None

            if is_ending:
                # Generate or fetch summary
                try:
                    summary = await ai_handler.generate_conversation_summary(conversation_history, contact)
                except Exception as e:
                    logger.error(f"Failed to generate conversation summary: {e}")
                    summary = None

                if config.owner_whatsapp_number:
                    # Build a defensive, concise notification using available data
                    meeting_info = ''
                    try:
                        # Try to infer meeting details from summary or conversation
                        if summary and summary.key_topics:
                            meeting_info = '\n'.join([f"- {t}" for t in summary.key_topics[:3]])
                    except Exception:
                        meeting_info = ''

                    notification_lines = [
                        "*New Contact Interaction Summary*",
                        f"Contact: {contact.name or contact.phone_number}",
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "",
                        "*Priority:* " + (priority.value if priority else 'UNKNOWN'),
                    ]

                    if meeting_info:
                        notification_lines += ["", "*Key Points:*", meeting_info]

                    if summary and summary.summary:
                        notification_lines += ["", "*Conversation Summary:*", summary.summary]

                    if summary and summary.action_items:
                        notification_lines += ["", "*Action Items:*"]
                        notification_lines += [f"- {item}" for item in summary.action_items]
                    else:
                        notification_lines += ["", "*Action Items:*", "- No explicit action items detected."]

                    notification_lines += ["", "Reason: " + (reason or 'AI detection')]

                    notification = "\n".join(notification_lines)

                    try:
                        async with EnhancedWhatsAppClient() as client:
                            await client.send_message(config.owner_whatsapp_number, notification)
                            logger.info("Sent conversation summary to owner")
                    except Exception as e:
                        logger.error(f"Failed to send summary to owner: {e}")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Message processed in {processing_time:.2f}ms for {contact.phone_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_message_data(self, webhook_data: Dict) -> Optional[Dict]:
        """Extract message data from webhook payload"""
        
        try:
            entry = webhook_data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            
            # Skip status updates
            if value.get("statuses"):
                return None
            
            messages = value.get("messages", [])
            if not messages:
                return None
            
            message = messages[0]
            
            # Extract content based on message type
            message_type = message.get("type", "text")
            content = ""
            
            if message_type == "text":
                content = message.get("text", {}).get("body", "")
            elif message_type == "image":
                content = f"[Image: {message.get('image', {}).get('caption', 'No caption')}]"
            elif message_type == "audio":
                content = "[Audio message]"
            elif message_type == "video":
                content = f"[Video: {message.get('video', {}).get('caption', 'No caption')}]"
            elif message_type == "document":
                doc_name = message.get('document', {}).get('filename', 'Unknown')
                content = f"[Document: {doc_name}]"
            else:
                content = f"[{message_type.upper()} message]"
            
            return {
                "message_id": message.get("id", str(uuid.uuid4())),
                "phone_number": message.get("from"),
                "message_type": message_type,
                "content": content,
                "raw_data": message
            }
            
        except Exception as e:
            logger.error(f"Failed to extract message data: {e}")
            return None

    async def _resolve_style(self, contact: ContactInfo, analysis: MessageAnalysis, message_text: str) -> Dict:
        """Resolve response styling preferences.

        Returns a dict with keys: language ('en'|'sw'), formality ('formal'|'informal'), verbosity ('short'|'normal'|'detailed').
        Uses contact preferences (if any), analysis.language, and simple heuristics (greeting->informal, business->formal).
        """
        try:
            lang = getattr(analysis, 'language', 'en')
            # Default formality: VIPs get slightly more formal
            formality = 'formal' if contact.is_vip else 'informal'

            # If analysis shows high business_relevance, prefer formal
            try:
                if getattr(analysis, 'business_relevance', 0) > 0.7:
                    formality = 'formal'
            except Exception:
                pass

            # Verbosity heuristics
            verbosity = 'normal'
            try:
                if getattr(analysis, 'urgency', 0) > 0.8:
                    verbosity = 'short'
                elif getattr(analysis, 'business_relevance', 0) > 0.8:
                    verbosity = 'detailed'
            except Exception:
                pass

            # Greeting messages should be friendly/informal by default
            if any(g in (message_text or '').lower() for g in ['habari', 'mambo', 'hello', 'hi', 'hey']):
                formality = 'informal'

            return {'language': lang, 'formality': formality, 'verbosity': verbosity}
        except Exception as e:
            logger.error(f"_resolve_style failed: {e}")
            return {'language': getattr(analysis, 'language', 'en'), 'formality': 'informal', 'verbosity': 'normal'}
    
    def _validate_message(self, message_data: Dict) -> bool:
        """Validate message for processing"""
        
        # Check for duplicate
        if cache_manager.is_duplicate_message(message_data['message_id']):
            logger.info(f"Skipping duplicate message: {message_data['message_id']}")
            return False
        
        # Check rate limits
        phone_number = message_data['phone_number']
        rate_key = f"rate:{phone_number}:{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        
        if not cache_manager.check_rate_limit(rate_key, config.max_messages_per_minute):
            logger.warning(f"Rate limit exceeded for {phone_number}")
            return False
        
        return True
    
    async def _save_message(self, message: EnhancedMessage):
        """Save message to database (delegates to services.persistence.save_message)"""
        try:
            from services.persistence import save_message
            await save_message(message, db_manager, cache_manager)
        except Exception as e:
            logger.error(f"_save_message delegation failed: {e}")
    
    async def _update_entities(self, conversation_id: str, contact: ContactInfo, priority: Priority, category: Category):
        """Update conversation and contact entities (delegates to services.persistence.update_entities)"""
        try:
            from services.persistence import update_entities
            await update_entities(
                conversation_id,
                contact,
                priority,
                category,
                db_manager,
                cache_manager,
                self.contact_manager
            )
        except Exception as e:
            logger.error(f"_update_entities delegation failed: {e}")

# Initialize processor
message_processor = MessageProcessor()

# FastAPI App with Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("Starting WhatsApp AI Agent...")
    
    try:
        await db_manager.initialize()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        # Don't raise - continue with limited functionality
    
    # Start background tasks
    setup_background_tasks()
    logger.info("✅ Background tasks started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down WhatsApp AI Agent...")
    
    # Close database connections
    try:
        await db_manager.close()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database connections: {e}")

app = FastAPI(
    title="Advanced WhatsApp AI Agent",
    description="Professional AI-powered WhatsApp business agent with advanced conversation management",
    version="2.0.0",
    lifespan=lifespan
)

# Security Middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure properly in production
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background Tasks Setup
def setup_background_tasks():
    """Setup scheduled background tasks"""
    
    try:
        @aiocron.crontab('0 */6 * * *')  # Every 6 hours
        async def cleanup_old_data():
            """Clean up old data and optimize database"""
            try:
                try:
                    from services.persistence import cleanup_old_messages
                    deleted = await cleanup_old_messages(db_manager, days=30, exclude_priority=Priority.CRITICAL.value)
                    logger.info(f"Cleaned up {deleted} old messages")
                except Exception as e:
                    logger.error(f"Cleanup delegation failed: {e}")
            
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
        
        logger.info("Background tasks scheduled successfully")
    except Exception as e:
        logger.error(f"Failed to setup background tasks: {e}")

# Main API Routes
@app.get("/", tags=["Health"])
async def health_check():
    """Enhanced health check with system status"""
    
    status = {
        "service": "Advanced WhatsApp AI Agent",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": "unknown",
            "cache": "unknown", 
            "ai": "unknown",
            "whatsapp_api": "unknown"
        }
    }
    
    # Check database
    try:
        if db_manager.client is not None:
            await asyncio.to_thread(db_manager.client.admin.command, 'ping')
            status["components"]["database"] = "healthy"
        else:
            status["components"]["database"] = "not_connected"
            status["status"] = "degraded"
    except Exception as e:
        status["components"]["database"] = "unhealthy"
        status["status"] = "degraded"
    
    # Check cache
    try:
        if db_manager.redis_client is not None:
            await db_manager.redis_client.ping()
            status["components"]["cache"] = "healthy"
        else:
            status["components"]["cache"] = "memory_only"
    except:
        status["components"]["cache"] = "unhealthy"
    
    # Check AI service
    try:
        if ai_handler.ai_available:
            status["components"]["ai"] = "configured"
        else:
            status["components"]["ai"] = "not_configured"
            status["status"] = "degraded"
    except:
        status["components"]["ai"] = "error"
    
    # Check WhatsApp API
    try:
        if config.whatsapp_access_token and config.whatsapp_phone_number_id:
            status["components"]["whatsapp_api"] = "configured"
        else:
            status["components"]["whatsapp_api"] = "not_configured"
            status["status"] = "degraded"
    except:
        status["components"]["whatsapp_api"] = "error"
    
    return status

@app.get("/api/webhook", tags=["WhatsApp"])
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"), 
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    """WhatsApp webhook verification with enhanced logging"""
    
    logger.info("🔍 Webhook verification requested")
    logger.info(f"Mode: {hub_mode}, Token: {hub_verify_token[:10]}...")
    
    if hub_mode == "subscribe" and hub_verify_token == config.webhook_verify_token:
        logger.info("✅ Webhook verification successful")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.error("❌ Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/api/webhook", tags=["WhatsApp"])
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Enhanced webhook handler with comprehensive processing"""
    
    try:
        # Rate limiting by IP
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"ip_rate:{client_ip}:{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        
        if not cache_manager.check_rate_limit(rate_key, config.max_requests_per_ip):
            logger.warning(f"IP rate limit exceeded: {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Parse webhook data
        try:
            webhook_data = await request.json()
        except Exception as e:
            logger.error(f"Invalid webhook payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Quick response for status updates
        entry = webhook_data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        
        if value.get("statuses"):
            logger.debug("Status update received")
            return JSONResponse({"status": "success"})
        
        # Log incoming message (minimal for performance)
        if value.get("messages"):
            msg = value["messages"][0]
            logger.info(f"📨 Message from {msg.get('from', 'unknown')[:4]}****")
        
        # Process in background
        # Lightweight dedup: if we've already seen this incoming message id, avoid scheduling again
        try:
            incoming_msgs = value.get("messages", [])
            if incoming_msgs:
                incoming_id = incoming_msgs[0].get("id") or hashlib.sha256(json.dumps(incoming_msgs[0], sort_keys=True).encode()).hexdigest()
                # Use a separate, short-lived cache key for webhook scheduling to avoid
                # interfering with the processor-level `message_dedup` which should be
                # the single source of truth for whether a message was already processed.
                webhook_key = f"webhook_schedule:{incoming_id}"
                try:
                    seen = await cache_manager.get(webhook_key)
                except Exception:
                    seen = None

                if seen:
                    logger.info(f"Duplicate webhook detected for {incoming_id}, ignoring")
                    return JSONResponse({"status": "success", "message": "duplicate_ignored"})

                # Mark scheduling key to avoid re-scheduling within a short window
                try:
                    await cache_manager.set(webhook_key, True, ttl=60)
                except Exception:
                    # Fallback to in-memory cache mark if async set fails
                    try:
                        cache_manager.memory_cache[webhook_key] = True
                    except Exception:
                        pass
        except Exception:
            pass

        background_tasks.add_task(message_processor.process_message, webhook_data)
        
        return JSONResponse({"status": "success", "message": "Processing"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": "Internal server error"}, status_code=500)

@app.get("/api/conversations", tags=["Conversations"])
async def get_conversations(
    limit: int = Query(50, le=200),
    priority: Optional[Priority] = None,
    category: Optional[Category] = None,
    status: Optional[ConversationStatus] = None,
    phone_number: Optional[str] = None
):
    """Get conversations with advanced filtering"""
    
    try:
        # Delegate conversation querying and formatting to persistence layer
        from services.persistence import query_conversations
        conversations = await query_conversations(
            db_manager,
            limit=limit,
            priority=(priority.value if priority else None),
            category=(category.value if category else None),
            status=(status.value if status else None),
            phone_number=phone_number
        )

        return {
            "conversations": conversations,
            "total": len(conversations),
            "filters_applied": {
                "priority": priority.value if priority else None,
                "category": category.value if category else None,
                "status": status.value if status else None
            }
        }
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return {"conversations": [], "error": str(e)}


@app.get("/api/personas", tags=["Personas"])
async def list_personas():
    """List available personas and the currently selected one"""
    try:
        personas = persona_manager.list_personas()
        return {"personas": personas, "current": persona_manager.current}
    except Exception as e:
        logger.error(f"Failed to list personas: {e}")
        raise HTTPException(500, "Failed to list personas")


@app.post("/api/personas/select", tags=["Personas"])
async def select_persona(name: str):
    """Select an existing persona by name (runtime switch)"""
    try:
        persona_manager.select_persona(name)
        # Rebuild AI system prompt to pick up the new persona immediately
        try:
            ai_handler.system_prompt = ai_handler._build_system_prompt()
        except Exception:
            pass
        return {"status": "ok", "selected": name}
    except KeyError:
        raise HTTPException(404, f"Persona '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to select persona {name}: {e}")
        raise HTTPException(500, "Failed to select persona")


@app.post("/api/personas", tags=["Personas"])
async def add_persona(name: str, description: str, system_prompt: str):
    """Add a new persona safely and make it available"""
    try:
        ok = persona_manager.add_persona(name, description, system_prompt)
        if not ok:
            raise HTTPException(500, "Failed to add persona")
        return {"status": "ok", "added": name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add persona {name}: {e}")
        raise HTTPException(500, "Failed to add persona")

@app.post("/api/messages/send", tags=["Messages"])
async def send_message(
    phone_number: str,
    message: str,
    message_type: str = "text"
):
    """Send a message via WhatsApp API"""
    
    try:
        # Validate inputs
        if not phone_number or not message:
            raise HTTPException(400, "Phone number and message are required")
        
        if len(message) > 4096:
            raise HTTPException(400, "Message too long (max 4096 characters)")
        
        # Send message
        async with EnhancedWhatsAppClient() as client:
            result = await client.send_message(phone_number, message, message_type)
        
        return {
            "status": "success",
            "message_id": result.get("messages", [{}])[0].get("id"),
            "whatsapp_response": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send message error: {e}")
        raise HTTPException(500, str(e))

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )

# Main application entry point
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Advanced WhatsApp AI Agent")
    
    # Production-ready uvicorn configuration
    uvicorn.run(
        app,  # Fixed: use app directly instead of string
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        access_log=True,
        reload=os.getenv("ENVIRONMENT", "production") == "development"
    )