from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PersonaManager:
    def __init__(self, personas_dir: str = "personas", default_prompt: Optional[str] = None):
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.personas = {}
        self.current = None
        
        # Load default persona from environment or parameter
        if default_prompt:
            self.current = {"name": "default", "system_prompt": default_prompt}
        
        self.load_personas()
        
        # Auto-select persona from environment
        agent_persona = os.getenv("AGENT_PERSONA")
        if agent_persona and agent_persona in self.personas:
            self.select_persona(agent_persona)
            logger.info(f"Auto-selected persona: {agent_persona}")

    def load_personas(self):
        """Load all persona files from the personas directory"""
        try:
            for p in self.personas_dir.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    # Validate required fields - description is now the primary prompt field
                    if not data.get("name"):
                        logger.warning(f"Invalid persona file {p}: missing name")
                        continue
                    
                    # Ensure description exists for AI prompting, fallback to system_prompt
                    if not data.get("description") and not data.get("system_prompt"):
                        logger.warning(f"Invalid persona file {p}: missing both description and system_prompt")
                        continue
                    
                    self.personas[data["name"]] = data
                    logger.debug(f"Loaded persona: {data['name']}")
                except Exception as e:
                    logger.error(f"Failed to load persona file {p}: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.personas)} personas: {list(self.personas.keys())}")
        except Exception as e:
            logger.error(f"Failed to load personas from {self.personas_dir}: {e}")

    def list_personas(self) -> List[str]:
        """Get list of available persona names"""
        return list(self.personas.keys())

    def get_persona_details(self, name: str) -> Optional[Dict[str, Any]]:
        """Get full details of a specific persona"""
        return self.personas.get(name)

    def get_all_personas(self) -> Dict[str, Dict[str, Any]]:
        """Get all personas with their details"""
        return self.personas.copy()

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt of the current persona - uses description as prompt for better persona-specific responses"""
        if self.current:
            # Use description as the AI prompt for more relevant persona responses
            description = self.current.get("description")
            if description:
                return description
            # Fallback to system_prompt if description is not available
            return self.current.get("system_prompt")
        return None

    def get_current_persona(self) -> Optional[Dict[str, Any]]:
        """Get the current persona details"""
        return self.current

    def select_persona(self, name: str) -> bool:
        """Select a persona by name"""
        if name in self.personas:
            self.current = self.personas[name].copy()
            logger.info(f"Selected persona: {name}")
            return True
        logger.warning(f"Persona not found: {name}")
        return False

    def add_persona(self, name: str, system_prompt: str, description: str = "", 
                   personality_traits: Optional[List[str]] = None,
                   capabilities: Optional[List[str]] = None,
                   tone: str = "professional",
                   language: str = "English",
                   response_style: str = "concise",
                   context_awareness: bool = True,
                   **kwargs) -> bool:
        """Add a new persona with comprehensive configuration"""
        try:
            data = {
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
                "personality_traits": personality_traits or [],
                "capabilities": capabilities or [],
                "tone": tone,
                "language": language,
                "response_style": response_style,
                "context_awareness": context_awareness,
                "created_at": datetime.now().isoformat(),
                **kwargs  # Allow additional custom fields
            }
            
            # Save to file
            filepath = self.personas_dir / f"{name}.json"
            filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Add to in-memory personas
            self.personas[name] = data
            
            logger.info(f"Added new persona: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add persona {name}: {e}")
            return False

    def update_persona(self, name: str, **updates) -> bool:
        """Update an existing persona"""
        if name not in self.personas:
            logger.warning(f"Cannot update non-existent persona: {name}")
            return False
        
        try:
            # Update in-memory persona
            self.personas[name].update(updates)
            self.personas[name]["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            filepath = self.personas_dir / f"{name}.json"
            filepath.write_text(
                json.dumps(self.personas[name], indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
            
            # Update current if it's the active persona
            if self.current and self.current.get("name") == name:
                self.current = self.personas[name].copy()
            
            logger.info(f"Updated persona: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update persona {name}: {e}")
            return False

    def delete_persona(self, name: str) -> bool:
        """Delete a persona"""
        if name not in self.personas:
            logger.warning(f"Cannot delete non-existent persona: {name}")
            return False
        
        try:
            # Remove file
            filepath = self.personas_dir / f"{name}.json"
            if filepath.exists():
                filepath.unlink()
            
            # Remove from memory
            del self.personas[name]
            
            # Clear current if it was the deleted persona
            if self.current and self.current.get("name") == name:
                self.current = None
            
            logger.info(f"Deleted persona: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete persona {name}: {e}")
            return False

    def reload_personas(self):
        """Reload all personas from disk"""
        self.personas.clear()
        self.load_personas()
        logger.info("Reloaded all personas from disk")

    def get_persona_stats(self) -> Dict[str, Any]:
        """Get statistics about personas"""
        return {
            "total_personas": len(self.personas),
            "current_persona": self.current.get("name") if self.current else None,
            "available_personas": self.list_personas(),
            "personas_directory": str(self.personas_dir)
        }
    
    def get_dynamic_persona_prompt(self, user_emotion: str = "neutral", 
                                 context_type: str = "general") -> Optional[str]:
        """Get a dynamic persona prompt that adapts to user emotion and context with enhanced instructions"""
        if not self.current:
            return None
        
        # Get base prompt
        base_prompt = self.get_system_prompt()
        if not base_prompt:
            return None
        
        # Enhanced emotion-specific instructions with more nuanced guidance
        emotion_instructions = {
            "frustrated": """
The user appears frustrated. Respond with increased empathy and a focus on problem-solving. Be patient and offer clear, actionable solutions.
- Acknowledge their frustration without being overly sympathetic
- Focus on resolving the specific issue they're facing
- Provide step-by-step guidance if applicable
- Avoid jargon or complex explanations
- Offer alternatives if the primary solution isn't working
""".strip(),
            "happy": """
The user appears happy and positive. Respond with enthusiasm and maintain a friendly, upbeat tone.
- Match their positive energy
- Be engaging and personable
- Celebrate their successes when relevant
- Keep responses concise but warm
- Suggest helpful ideas that build on their positive mood
""".strip(),
            "urgent": """
The user has an urgent matter. Respond quickly and efficiently, prioritizing their immediate needs.
- Get straight to the point
- Focus on the most critical information first
- Provide immediate next steps
- Avoid unnecessary details or pleasantries
- If you can't resolve immediately, explain what you'll do next
""".strip(),
            "confused": """
The user seems confused. Provide clear, step-by-step explanations and avoid complex terminology.
- Break down complex concepts into simple terms
- Use analogies or examples when helpful
- Check for understanding periodically
- Avoid technical jargon unless necessary
- Offer to explain in a different way if needed
""".strip(),
            "sad": """
The user appears sad or upset. Respond with compassion and offer supportive, gentle assistance.
- Show empathy without being overly emotional
- Offer practical support and solutions
- Avoid minimizing their feelings
- Provide reassurance when appropriate
- Suggest resources or next steps that might help
""".strip(),
            "angry": """
The user is angry. Stay calm, acknowledge their frustration, and focus on de-escalation and resolution.
- Don't take their anger personally
- Acknowledge their feelings without agreeing or disagreeing
- Focus on facts and solutions
- Avoid defensive language
- Offer concrete steps to address their concerns
""".strip(),
            "curious": """
The user is curious and asking questions. Be informative and encourage their exploration with detailed, helpful responses.
- Provide comprehensive answers to their questions
- Offer additional relevant information
- Encourage further exploration with related topics
- Use examples to illustrate concepts
- Be patient with follow-up questions
""".strip(),
            "neutral": """
Maintain a professional and helpful tone while being responsive to the user's specific needs.
- Be clear and concise
- Provide accurate information
- Adapt tone based on the content of their message
- Offer assistance proactively when appropriate
- Stay focused on their goals
""".strip()
        }
        
        # Enhanced context-specific instructions with detailed guidance
        context_instructions = {
            "technical": """
Provide detailed technical explanations and use appropriate terminology.
- Use precise technical terms when relevant
- Explain complex concepts with examples
- Provide code snippets or technical specifications when helpful
- Link to documentation or resources when available
- Be thorough but avoid unnecessary complexity
""".strip(),
            "business": """
Maintain a professional business tone with clear, concise communication.
- Focus on outcomes and results
- Use business-appropriate language
- Be direct and efficient
- Include relevant metrics or data when applicable
- Consider timelines and deadlines in your responses
""".strip(),
            "support": """
Focus on customer service principles, being helpful, patient, and solution-oriented.
- Prioritize resolving the user's issue
- Be patient and thorough in explanations
- Follow up on previous interactions when relevant
- Offer multiple solutions when possible
- Document key points for future reference
""".strip(),
            "creative": """
Be imaginative and open to creative ideas and solutions.
- Encourage brainstorming and exploration
- Suggest innovative approaches
- Build on their ideas rather than redirecting
- Use creative language and metaphors when appropriate
- Inspire without being overly abstract
""".strip(),
            "general": """
Maintain a balanced, helpful approach suitable for general conversation.
- Adapt your tone to match the conversation flow
- Be personable while remaining professional
- Offer assistance proactively
- Stay focused on providing value
- Keep responses natural and conversational
""".strip()
        }
        
        # Build dynamic prompt
        dynamic_prompt = base_prompt
        
        # Add emotion-specific instructions
        if user_emotion in emotion_instructions:
            dynamic_prompt += f"\n\n{emotion_instructions[user_emotion]}"
        
        # Add context-specific instructions
        if context_type in context_instructions:
            dynamic_prompt += f"\n\n{context_instructions[context_type]}"
        
        # Add general adaptive behavior instructions
        dynamic_prompt += """

ADAPTIVE BEHAVIOR GUIDELINES:
- Monitor the conversation flow and adjust your approach as needed
- If the user changes topics or emotions, adapt accordingly
- Maintain consistency with previously stated facts or commitments
- When uncertain, ask clarifying questions rather than making assumptions
- Balance being helpful with respecting user boundaries
""".strip()
        
        return dynamic_prompt
    
    def detect_user_emotion(self, text: str) -> str:
        """Detect user emotion from text with enhanced detection using sentiment analysis"""
        text_lower = text.lower().strip()
        
        # Enhanced emotion detection with sentiment scoring
        emotion_scores = {
            "frustrated": 0,
            "happy": 0,
            "urgent": 0,
            "confused": 0,
            "sad": 0,
            "angry": 0,
            "curious": 0,
            "neutral": 0
        }
        
        # Emotion detection keywords with weights
        emotion_keywords = {
            "frustrated": [
                ("frustrated", 2), ("annoyed", 2), ("irritated", 2), ("upset", 1), 
                ("disappointed", 1), ("angry", 1), ("mad", 1), ("furious", 2),
                ("why won't", 2), ("doesn't work", 2), ("can't get", 1), ("stuck", 1)
            ],
            "happy": [
                ("happy", 2), ("excited", 2), ("pleased", 1), ("delighted", 2), 
                ("joyful", 2), ("thrilled", 2), ("great", 1), ("awesome", 2),
                ("love it", 2), ("wonderful", 2), ("fantastic", 2), ("brilliant", 2)
            ],
            "urgent": [
                ("urgent", 3), ("asap", 3), ("immediately", 3), ("now", 2), 
                ("quick", 1), ("hurry", 2), ("rush", 2), ("emergency", 3),
                ("need this", 2), ("right away", 2), ("stat", 3)
            ],
            "confused": [
                ("confused", 2), ("don't understand", 3), ("unclear", 2), ("not sure", 2), 
                ("help me", 2), ("explain", 2), ("what does", 1), ("how does", 1),
                ("can you explain", 2), ("i'm lost", 3), ("i don't get it", 3)
            ],
            "sad": [
                ("sad", 2), ("depressed", 2), ("unhappy", 2), ("disappointed", 1), 
                ("sorry to hear", 1), ("unfortunate", 1), ("upset", 1), ("worried", 2)
            ],
            "angry": [
                ("angry", 3), ("mad", 2), ("furious", 3), ("outraged", 3), ("livid", 3),
                ("this is unacceptable", 3), ("ridiculous", 2), ("absurd", 2)
            ],
            "curious": [
                ("curious", 2), ("wondering", 2), ("interested", 2), ("question", 1), 
                ("how does", 1), ("tell me about", 2), ("can you tell", 1), ("what is", 1),
                ("how do i", 1), ("i want to know", 2)
            ]
        }
        
        # Calculate emotion scores
        for emotion, keywords in emotion_keywords.items():
            for keyword, weight in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += weight
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Return neutral if no strong emotion detected
        if emotion_scores[dominant_emotion] == 0:
            return "neutral"
        
        return dominant_emotion
    
    def detect_context_type(self, text: str) -> str:
        """Detect conversation context type from text with enhanced detection using context scoring"""
        text_lower = text.lower().strip()
        
        # Enhanced context detection with scoring
        context_scores = {
            "technical": 0,
            "business": 0,
            "support": 0,
            "creative": 0,
            "general": 0
        }
        
        # Context detection keywords with weights
        context_keywords = {
            "technical": [
                ("api", 2), ("code", 2), ("function", 1), ("error", 2), ("debug", 2), 
                ("technical", 2), ("programming", 2), ("software", 1), ("bug", 2),
                ("script", 1), ("algorithm", 2), ("database", 1), ("framework", 1)
            ],
            "business": [
                ("meeting", 2), ("schedule", 2), ("appointment", 2), ("contract", 2), 
                ("deal", 2), ("business", 2), ("client", 1), ("customer", 1),
                ("proposal", 2), ("budget", 2), ("revenue", 1), ("sales", 1)
            ],
            "support": [
                ("help", 2), ("issue", 2), ("problem", 2), ("broken", 2), 
                ("not working", 3), ("support", 2), ("assistance", 2), ("troubleshoot", 2),
                ("fix", 1), ("resolve", 1), ("bug", 2), ("error", 2)
            ],
            "creative": [
                ("idea", 2), ("creative", 2), ("design", 2), ("art", 1), 
                ("music", 1), ("write", 1), ("story", 1), ("imagine", 1),
                ("brainstorm", 2), ("innovative", 2), ("concept", 1)
            ]
        }
        
        # Calculate context scores
        for context, keywords in context_keywords.items():
            for keyword, weight in keywords:
                if keyword in text_lower:
                    context_scores[context] += weight
        
        # Determine dominant context
        dominant_context = max(context_scores.items(), key=lambda x: x[1])[0]
        
        # Return general if no strong context detected
        if context_scores[dominant_context] == 0:
            return "general"
        
        return dominant_context
