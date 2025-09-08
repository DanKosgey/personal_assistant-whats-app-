from typing import Dict, Any, List
from ..ai import AdvancedAIHandler
from ..clients.whatsapp import EnhancedWhatsAppClient, WhatsAppAPIError
import logging
import random
from datetime import datetime
from .personas import (
    PERSONAL_ASSISTANT_PERSONA,
    GREETING_TEMPLATES,
    TASK_ACKNOWLEDGMENTS,
    ERROR_MESSAGES,
    format_response
)
from .memory import MemoryManager
from .persistence import get_or_create_contact, update_contact
from ..db import db_manager
import re
import os
import asyncio
from .context import load_context
from ..prompts import build_agent_instruction_prompt, build_summary_prompt

logger = logging.getLogger(__name__)


class MessageProcessor:
    def __init__(self, ai: AdvancedAIHandler, whatsapp: EnhancedWhatsAppClient):
        self.ai = ai
        self.whatsapp = whatsapp
        self.memory = MemoryManager()

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format recent context for AI prompt"""
        if not context:
            return ""
        
        context_str = "\nPrevious conversation context:\n"
        for ctx in context[-3:]:  # Last 3 interactions
            context_str += f"User: {ctx['message']}\n"
            context_str += f"Assistant: {ctx['response']}\n"
        return context_str

    async def _enhance_prompt(self, sender: str, text: str) -> str:
        """Enhance user prompt with office assistant context and conversation history"""
        memory = await self.memory.get_user_memory(sender)
        recent_context = await self.memory.get_recent_context(sender)

        # Load global assistant/company context for persona-aware prompts
        ctx = load_context()
        assistant_name = ctx.get('assistant_name', 'Secretary')
        company = ctx.get('company_name') or memory.preferences.get('company') or 'Sir Williams - Data Science'

        return build_agent_instruction_prompt(
            sender=sender,
            text=text,
            assistant_name=assistant_name,
            company_name=company,
            user_preferences=memory.preferences,
            recent_context=recent_context,
        )

    def _is_greeting(self, text: str) -> bool:
        """Check if the message is a greeting"""
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        return text.lower().strip().rstrip("?!.") in greetings

    async def process(self, sender: str, text: str) -> Dict[str, Any]:
        logger.info("Processing message from %s: %s", sender, text)
        try:
            # First verify AI handler is ready
            if not self.ai.ai_available:
                logger.error("AI handler not available: %s", 
                    f"Next retry at {self.ai.next_retry_time}" if self.ai.next_retry_time else "No retry time set")
                await self.whatsapp.send_message(sender, ERROR_MESSAGES["technical"])
                raise RuntimeError("AI service unavailable")

            # Analyze and generate a response
            try:
                analysis = await self.ai.analyze(text)
                logger.info("Message analysis: %s", analysis)
            except Exception as e:
                logger.error("Analysis failed: %s", e)
                analysis = {"error": str(e)}

            # Handle greetings with persona-specific responses
            if self._is_greeting(text):
                response_text = random.choice(GREETING_TEMPLATES)
                send_result = await self.whatsapp.send_message(sender, response_text)
                return {
                    "analysis": analysis,
                    "generated": {"text": response_text, "provider": "persona"},
                    "send": send_result
                }

            # Direct question: who are you? -> respond with templated assistant identity
            _re = __import__('re')
            if _re.search(r"\bwho\s+are\s+you\b", text, flags=_re.IGNORECASE):
                ctx = load_context()
                mem = await self.memory.get_user_memory(sender)
                assistant_name = ctx.get('assistant_name') or 'Secretary'
                company = ctx.get('company_name') or mem.preferences.get('company') or 'Sir Williams - Data Science'
                owner = ctx.get('owner_name') or mem.preferences.get('owner_name') or 'Sir Williams'
                short_desc = ctx.get('short_description', '')
                signature = ctx.get('default_signature', '{{assistant_name}}\n{{company_name}}')
                signature_filled = signature.replace('{{assistant_name}}', assistant_name).replace('{{company_name}}', company)
                # Clear, concise secretary-style identity
                response_text = (
                    f"Hello,\n\nMy name is {assistant_name}, and I am the personal secretary for {owner} ({company}).\n"
                    f"{short_desc}\n\nHow can I assist you today?\n\nThank you,\n{signature_filled}"
                )
                send_result = await self.whatsapp.send_message(sender, response_text)
                return {
                    "analysis": analysis,
                    "generated": {"text": response_text, "provider": "persona"},
                    "send": send_result
                }

            # Generate AI response with enhanced prompt
            try:
                enhanced_prompt = await self._enhance_prompt(sender, text)
                gen = await self.ai.generate(enhanced_prompt)
                logger.info("Generated AI response: %s", gen)
            except Exception as e:
                logger.error("Generation failed: %s", e)
                error_msg = ERROR_MESSAGES["technical"]
                await self.whatsapp.send_message(sender, error_msg)
                return {"error": "ai_generation_failed", "analysis": analysis}
            
            if not gen or not gen.get("text"):
                error_msg = ERROR_MESSAGES["technical"]
                await self.whatsapp.send_message(sender, error_msg)
                return {"error": "ai_generation_failed", "analysis": analysis}
                
            response_text = gen.get("text", "")
        except Exception as e:
            error_msg = "I apologize, but there was an error processing your message. Please try again in a few moments."
            logger.error("Error processing message: %s", str(e))
            try:
                await self.whatsapp.send_message(sender, error_msg)
            except Exception as send_error:
                logger.error("Failed to send error message: %s", str(send_error))
            raise

        # Clean provider tags added for debugging/dev (e.g. "[DEV_SMOKE reply to: ...]" or "[Gemini reply to: ...]")
        # Keep only the core reply text.
        if isinstance(response_text, str):
            # If the AI handler wrapped replies in square brackets with a provider prefix, strip it.
            # Examples handled:
            #  [DEV_SMOKE reply to: Hello]
            #  [Gemini reply to: Hello]
            import re

            # Strip any provider wrapper but keep the AI-generated response
            m = re.match(r"^\[(?:[A-Za-z0-9_\- ]+?) reply to: .*?\](.*?)$", response_text)
            if m and m.group(1):  # If we matched and have content after the wrapper
                response_text = m.group(1).strip()
            elif not response_text:  # Fallback if empty
                response_text = "I apologize, but I couldn't generate a response at the moment."

        try:
            # Store message and response in memory
            await self.memory.add_to_context(sender, text, response_text)

            # Simple entity extraction: phone numbers and emails
            phones = re.findall(r"\+?\d[\d\s\-()]{6,}\d", text)
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

            # Name heuristics: look for "my name is <Name>" or capitalized words near greeting
            name = None
            m = re.search(r"my name is\s+([A-Za-z ,.'-]{2,50})", text, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip().title()

            # Persist contact(s) — prefer phone numbers for dedupe; fallback to sender id
            contact_key = sender
            # If phone-like found, prefer the first one
            if phones:
                contact_key = re.sub(r"[^0-9+]", "", phones[0])

            contact_info = {
                "phone": contact_key,
                "name": name or None,
                "emails": emails,
                "source": "whatsapp",
                "first_contacted": datetime.utcnow().isoformat()
            }

            # Update in-memory contacts
            await self.memory.add_contact(sender, contact_info)

            # Try to persist to DB if available
            try:
                contact_doc = await get_or_create_contact(contact_key, db_manager, db_manager)
                # Merge fields and update
                updates = {}
                if name and not contact_doc.get('name'):
                    updates['name'] = name
                if emails and not contact_doc.get('email'):
                    updates['email'] = emails[0]
                if updates:
                    await update_contact(contact_key, updates, db_manager, db_manager)
            except Exception as e:
                logger.debug("Contact persistence skipped/failed: %s", e)

            # Update user preferences if explicitly mentioned
            lower = text.lower()
            if "my name is" in lower and name:
                await self.memory.set_preference(sender, "name", name)

            role = None
            for marker in ["my role is", "i am a", "i work as"]:
                if marker in lower:
                    role = lower.split(marker)[-1].strip().split(".\n")[0][:50].title()
                    await self.memory.set_preference(sender, "role", role)
                    break

            if "timezone" in lower:
                if "utc" in text.upper() or "gmt" in text.upper():
                    await self.memory.set_preference(sender, "timezone", text)

            if "work hours" in lower or "working hours" in lower:
                if ":" in text and "-" in text:
                    await self.memory.set_preference(sender, "work_hours", text)

            # End-of-conversation detection (simple heuristics)
            eoc_keywords = ["thank you", "thanks", "thankyou", "bye", "goodbye", "see you", "regards"]
            is_eoc = any(k in lower for k in eoc_keywords)
            if is_eoc:
                # Mark conversation ended in DB if possible
                try:
                    # Try to find an active conversation and mark it ended.
                    # Use getattr so static type checkers don't assume a specific API.
                    convs = None
                    get_col = getattr(db_manager, 'get_collection', None)
                    if callable(get_col):
                        try:
                            convs = get_col('conversations')
                        except Exception:
                            convs = None

                    if convs is None:
                        # Fallback: try to access a `db` attribute (motor/pymongo style)
                        db_attr = getattr(db_manager, 'db', None)
                        if db_attr is not None:
                            try:
                                convs = getattr(db_attr, 'conversations', None) or db_attr['conversations']
                            except Exception:
                                convs = None

                    if convs is not None:
                        try:
                            upd = getattr(convs, 'update_one', None)
                            if callable(upd):
                                await asyncio.to_thread(
                                    upd,
                                    {"phone_number": contact_key, "status": "active"},
                                    {"$set": {"status": "ended", "end_time": datetime.utcnow().isoformat()}},
                                    upsert=False
                                )
                        except Exception:
                            # Be forgiving if the DB driver uses a different API
                            pass
                except Exception:
                    pass

                # Optionally escalate to owner number — environment variable or memory preference
                owner_number = os.getenv('PA_OWNER_NUMBER') or (await self.memory.get_preference(sender, 'owner_number', None))
                if owner_number:
                    # Prepare conversation summary and suggested agenda/actions
                    recent_ctx = await self.memory.get_recent_context(sender)
                    # Build a brief human prompt for summarization
                    try:
                        # Create a compact transcript
                        transcript = "\n".join([
                            f"U: {c['message']}\nA: {c['response']}" for c in recent_ctx[-10:]
                        ]) if recent_ctx else text

                        summarization_prompt = build_summary_prompt(
                            contact_display=contact_info.get('name') or contact_key,
                            contact_phone=contact_key,
                            transcript=transcript,
                        )

                        summary_text = None
                        # Use AI to generate summary if available
                        if getattr(self.ai, 'ai_available', False):
                            try:
                                gen_sum = await self.ai.generate(summarization_prompt)
                                if gen_sum:
                                    gen_text = gen_sum.get('text') or gen_sum.get('message') or ''
                                    summary_text = str(gen_text).strip() if gen_text is not None else None
                            except Exception as e:
                                logger.debug("AI summarization failed: %s", e)

                        # Fallback summary
                        if not summary_text:
                            last_msgs = transcript if isinstance(transcript, str) else text
                            summary_text = (
                                f"Conversation ended with {contact_info.get('name') or contact_key}. "
                                f"Last messages: {last_msgs[:200]}. Suggested next steps: follow up to confirm details, propose meeting times, collect missing contact info."
                            )

                        owner_message = (
                            f"[PA Summary] Conversation ended with {contact_info.get('name') or contact_key}\n\n"
                            f"{summary_text}\n\n"
                            f"Contact: {contact_info.get('name') or 'Unknown'} ({contact_key})\n"
                        )

                        # Send summary to owner
                        try:
                            await self.whatsapp.send_message(owner_number, owner_message)
                        except Exception as e:
                            logger.debug("Failed to notify owner %s: %s", owner_number, e)

                        # Persist summary in conversations collection if available
                        try:
                            get_col = getattr(db_manager, 'get_collection', None)
                            convs = None
                            if callable(get_col):
                                convs = get_col('conversations')
                            if convs is None:
                                db_attr = getattr(db_manager, 'db', None)
                                if db_attr is not None:
                                    convs = getattr(db_attr, 'conversations', None) or db_attr['conversations']
                            if convs is not None:
                                upd = getattr(convs, 'update_one', None)
                                if callable(upd):
                                    await asyncio.to_thread(
                                        upd,
                                        {"phone_number": contact_key, "status": "ended"},
                                        {"$set": {"summary": summary_text, "status": "ended", "end_time": datetime.utcnow().isoformat()}},
                                        upsert=False,
                                    )
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug("Failed to prepare owner summary: %s", e)

            send_result = await self.whatsapp.send_message(sender, response_text)
        except WhatsAppAPIError as e:
            logger.error("WhatsApp send failed: %s", e)
            send_result = {"status": "failed", "error": str(e)}

        return {
            "analysis": analysis,
            "generated": gen,
            "send": send_result,
            "context_stored": True
        }
