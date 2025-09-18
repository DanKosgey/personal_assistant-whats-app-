"""
Intent Detection and Slot Capture System
Detects user intents and extracts relevant information from conversations
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntentDetector:
    """System for detecting user intents and capturing relevant slots"""
    
    def __init__(self):
        # Define intent patterns and slot extractors
        self.intent_patterns = {
            "meeting_request": {
                "keywords": [
                    "meeting", "appointment", "schedule.*meeting", "set.*meeting", 
                    "book.*meeting", "arrange.*meeting", "discuss", "call.*tomorrow",
                    "call.*today", "talk.*later", "meet.*up", "catch.*up"
                ],
                "slots": {
                    "date": r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2})",
                    "time": r"(\d{1,2}:\d{2}\s*(?:AM|PM)?|\d{1,2}\s*(?:AM|PM))",
                    "location": r"(office|home|zoom|teams|google meet|skype|call|phone|video)",
                    "topic": r"(?:about|regarding|concerning)\s+([^.!?]+)"
                }
            },
            "payment_request": {
                "keywords": [
                    "payment", "pay", "invoice", "bill", "cost", "price", "fee", 
                    "charge", "amount", "shilling", "dollar", "money", "transfer"
                ],
                "slots": {
                    "amount": r"(\d+(?:\.\d{2})?)\s*(?:shillings?|dollars?|\$|KES)",
                    "currency": r"(shilling|dollar|\$|KES|USD|KSH)",
                    "due_date": r"(?:due|by|before)\s+(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2})",
                    "invoice_number": r"(?:invoice|ref|reference)\s+#?([A-Z0-9-]+)"
                }
            },
            "information_request": {
                "keywords": [
                    "information", "info", "details", "question", "help", "assist",
                    "tell me", "what is", "how do", "can you", "could you", "please"
                ],
                "slots": {
                    "topic": r"(?:about|regarding|concerning)\s+([^.!?]+)",
                    "urgency": r"(urgent|asap|immediately|soon|whenever you can)"
                }
            },
            "problem_report": {
                "keywords": [
                    "problem", "issue", "broken", "error", "not working", "doesn't work",
                    "trouble", "difficulty", "stuck", "failed", "bug"
                ],
                "slots": {
                    "system": r"(?:with|on|in)\s+([^.!?]+)",
                    "urgency": r"(urgent|asap|immediately|critical)",
                    "impact": r"(?:affecting|impacting)\s+([^.!?]+)"
                }
            },
            "document_request": {
                "keywords": [
                    "document", "contract", "agreement", "sign", "paperwork", "file",
                    "report", "form", "application", "submit"
                ],
                "slots": {
                    "document_type": r"(?:contract|agreement|report|form|application|document)\s+([^.!?]+)?",
                    "due_date": r"(?:due|by|before)\s+(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2})",
                    "urgency": r"(urgent|asap|immediately)"
                }
            },
            "follow_up": {
                "keywords": [
                    "follow.*up", "update", "status", "check.*status", "progress",
                    "how.*going", "any news", "what.*happening"
                ],
                "slots": {
                    "topic": r"(?:about|regarding|concerning)\s+([^.!?]+)",
                    "urgency": r"(urgent|asap|immediately|soon)"
                }
            },
            "cancellation": {
                "keywords": [
                    "cancel", "refund", "return", "terminate", "stop", "end", 
                    "discontinue", "unsubscribe"
                ],
                "slots": {
                    "reason": r"(?:because|due to|reason)\s+([^.!?]+)",
                    "urgency": r"(urgent|asap|immediately)"
                }
            },
            "feedback": {
                "keywords": [
                    "feedback", "complaint", "suggestion", "review", "opinion",
                    "comment", "improve", "better"
                ],
                "slots": {
                    "topic": r"(?:about|regarding|concerning)\s+([^.!?]+)",
                    "sentiment": r"(good|bad|great|terrible|excellent|poor|awesome)"
                }
            }
        }
        
        # Confidence weights for different pattern types
        self.confidence_weights = {
            "strong_keyword": 0.4,
            "medium_keyword": 0.25,
            "slot_match": 0.2,
            "context_match": 0.15
        }
    
    def detect_intent(self, text: str, context: List[Dict[str, Any]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Detect the primary intent from text and extract relevant slots
        
        Returns:
            Tuple of (intent, confidence, slots)
        """
        text_lower = text.lower()
        context_text = self._get_context_text(context)
        
        best_intent = "unknown"
        best_confidence = 0.0
        best_slots = {}
        
        # Check each intent pattern
        for intent_name, intent_data in self.intent_patterns.items():
            confidence, slots = self._evaluate_intent(text_lower, context_text, intent_data)
            
            if confidence > best_confidence:
                best_intent = intent_name
                best_confidence = confidence
                best_slots = slots
        
        # If confidence is low, check for general intents
        if best_confidence < 0.3:
            # Check for greeting
            greeting_patterns = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
            if any(pattern in text_lower for pattern in greeting_patterns):
                best_intent = "greeting"
                best_confidence = 0.8
                best_slots = {}
            
            # Check for goodbye
            goodbye_patterns = ["bye", "goodbye", "see you", "talk to you", "later"]
            if any(pattern in text_lower for pattern in goodbye_patterns):
                best_intent = "goodbye"
                best_confidence = 0.8
                best_slots = {}
        
        logger.debug(f"Detected intent: {best_intent} (confidence: {best_confidence:.2f})")
        return best_intent, best_confidence, best_slots
    
    def _evaluate_intent(self, text: str, context_text: str, intent_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate how well text matches an intent pattern"""
        confidence = 0.0
        slots = {}
        
        # Check for strong keywords
        strong_keywords = intent_data["keywords"]
        keyword_matches = 0
        for pattern in strong_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                keyword_matches += 1
                confidence += self.confidence_weights["strong_keyword"]
        
        # Extract slots
        slot_patterns = intent_data["slots"]
        slot_matches = 0
        for slot_name, pattern in slot_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                slots[slot_name] = match.group(1) if match.groups() else match.group(0)
                slot_matches += 1
                confidence += self.confidence_weights["slot_match"]
        
        # Check context for additional support
        if context_text:
            # Look for supporting keywords in context
            context_keywords = [kw for kw in strong_keywords if re.search(kw, context_text, re.IGNORECASE)]
            if context_keywords:
                confidence += min(len(context_keywords) * 0.05, self.confidence_weights["context_match"])
        
        # Normalize confidence (clamp to 0-1 range)
        confidence = min(confidence, 1.0)
        
        # Boost confidence for intents with both keywords and slots
        if keyword_matches > 0 and slot_matches > 0:
            confidence = min(confidence * 1.2, 1.0)
        
        return confidence, slots
    
    def _get_context_text(self, context: List[Dict[str, Any]]) -> str:
        """Extract text from conversation context"""
        if not context:
            return ""
        
        # Get recent messages (last 3 exchanges)
        recent_context = context[-3:] if len(context) > 3 else context
        context_text = " ".join([
            f"{msg.get('message', '')} {msg.get('response', '')}"
            for msg in recent_context
        ])
        
        return context_text.lower()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract common entities from text"""
        entities = defaultdict(list)
        
        # Extract dates
        date_patterns = [
            r"\b(today|tomorrow|yesterday)\b",
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dates"].extend(matches)
        
        # Extract times
        time_patterns = [
            r"\b(\d{1,2}:\d{2}\s*(?:AM|PM)?)\b",
            r"\b(\d{1,2}\s*(?:AM|PM))\b"
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["times"].extend(matches)
        
        # Extract monetary amounts
        money_patterns = [
            r"\b(\d+(?:\.\d{2})?)\s*(?:shillings?|dollars?|\$|KES|USD|KSH)\b",
            r"\b(?:shillings?|dollars?|\$|KES|USD|KSH)\s*(\d+(?:\.\d{2})?)\b"
        ]
        
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["amounts"].extend(matches)
        
        # Extract phone numbers
        phone_patterns = [
            r"\b(\+?\d{10,15})\b",
            r"\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b"
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            entities["phone_numbers"].extend(matches)
        
        # Extract email addresses
        email_pattern = r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
        matches = re.findall(email_pattern, text)
        entities["emails"].extend(matches)
        
        return dict(entities)
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of an intent"""
        descriptions = {
            "meeting_request": "Request for a meeting or appointment",
            "payment_request": "Request for payment or invoice information",
            "information_request": "Request for information or assistance",
            "problem_report": "Report of a problem or issue",
            "document_request": "Request for documents or contracts",
            "follow_up": "Follow-up on a previous conversation",
            "cancellation": "Request to cancel or terminate something",
            "feedback": "Providing feedback or complaints",
            "greeting": "Greetings or introductions",
            "goodbye": "Farewells or conversation endings",
            "unknown": "Uncategorized intent"
        }
        
        return descriptions.get(intent, "Unknown intent")
    
    def get_priority_level(self, intent: str, confidence: float, slots: Dict[str, Any]) -> str:
        """Determine priority level based on intent and context"""
        # High priority intents
        high_priority_intents = [
            "problem_report", "payment_request", "cancellation", 
            "urgent_meeting_request", "critical_issue"
        ]
        
        # Check for urgency indicators in slots
        urgency_indicators = ["urgent", "asap", "immediately", "critical", "emergency"]
        has_urgency = any(
            any(indicator in str(value).lower() for indicator in urgency_indicators)
            for value in slots.values()
        )
        
        # Check for high-value keywords
        high_value_keywords = [
            "contract", "agreement", "sign", "legal", "lawyer", 
            "doctor", "medical", "hospital", "emergency"
        ]
        has_high_value = any(
            any(keyword in str(value).lower() for keyword in high_value_keywords)
            for value in slots.values()
        )
        
        # Determine priority based on intent, confidence, and indicators
        if intent in high_priority_intents or has_urgency or has_high_value:
            return "HIGH"
        elif confidence >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"