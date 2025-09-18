"""
EOC (End of Conversation) Detector with Embedding Support
Implements embedding-based similarity detection for more accurate EOC detection
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EOCExample:
    """Example of EOC text with metadata"""
    id: str
    text: str
    embedding: List[float]
    confidence: float = 1.0
    category: str = "general"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class EOCEmbeddingDetector:
    """EOC detector using embedding similarity"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.eoc_examples: List[EOCExample] = []
        self._initialize_eoc_examples()
        
    def _initialize_eoc_examples(self):
        """Preload EOC examples with embeddings"""
        examples = [
            # General EOC phrases
            ("eoc_001", "Thank you for your help", "thanks"),
            ("eoc_002", "Thanks for the information", "thanks"),
            ("eoc_003", "Thank you so much", "thanks"),
            ("eoc_004", "Thanks a lot", "thanks"),
            ("eoc_005", "That's very helpful", "thanks"),
            ("eoc_006", "I appreciate your assistance", "thanks"),
            
            # Goodbye phrases
            ("eoc_007", "Goodbye", "goodbye"),
            ("eoc_008", "Bye for now", "goodbye"),
            ("eoc_009", "See you later", "goodbye"),
            ("eoc_010", "Take care", "goodbye"),
            ("eoc_011", "Have a great day", "goodbye"),
            ("eoc_012", "Talk to you soon", "goodbye"),
            
            # Meeting/conversation enders
            ("eoc_013", "That's all for now", "meeting_end"),
            ("eoc_014", "I think we've covered everything", "meeting_end"),
            ("eoc_015", "Let's wrap this up", "meeting_end"),
            ("eoc_016", "I'll follow up on that", "meeting_end"),
            ("eoc_017", "We can discuss this later", "meeting_end"),
            
            # Confirmation/acknowledgment
            ("eoc_018", "Sounds good", "confirmation"),
            ("eoc_019", "That works for me", "confirmation"),
            ("eoc_020", "Perfect", "confirmation"),
            ("eoc_021", "Great", "confirmation"),
            ("eoc_022", "Okay", "confirmation"),
            ("eoc_023", "Understood", "confirmation"),
        ]
        
        for example_id, text, category in examples:
            embedding = self._generate_embedding(text)
            example = EOCExample(
                id=example_id,
                text=text,
                embedding=embedding,
                category=category
            )
            self.eoc_examples.append(example)
            
        logger.info(f"Initialized EOC detector with {len(self.eoc_examples)} examples")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text (placeholder implementation)
        In a real implementation, this would call an actual embedding service
        """
        # Simple hash-based embedding for demonstration
        # In production, use a proper embedding model like SentenceTransformer
        import hashlib
        # Use a more sophisticated approach with multiple hashes
        hashes = []
        for i in range(16):
            # Create variations of the text for more diverse hashing
            modified_text = f"{text}_{i}"
            hash_val = hashlib.md5(modified_text.encode()).hexdigest()
            hashes.append(hash_val)
        
        # Convert hashes to a vector with more variation
        vector = []
        for hash_val in hashes:
            # Take 2 characters at a time and convert to float
            for j in range(0, len(hash_val), 2):
                if len(vector) >= 16:  # Limit vector size
                    break
                hex_pair = hash_val[j:j+2]
                # Convert hex to normalized float (0-1 range)
                float_val = float(int(hex_pair, 16)) / 255.0
                vector.append(float_val)
        
        # Ensure we have exactly 16 dimensions
        while len(vector) < 16:
            vector.append(0.0)
        vector = vector[:16]
        
        # Normalize vector
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        else:
            # If norm is 0, create a small random vector
            import random
            vector = [random.random() for _ in range(16)]
            norm = sum(v * v for v in vector) ** 0.5
            vector = [v / norm for v in vector]
            
        return vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def detect_eoc(self, text: str) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if text indicates end of conversation using embedding similarity
        
        Returns:
            Tuple of (is_eoc, confidence, example_id)
        """
        try:
            # Generate embedding for input text
            text_embedding = self._generate_embedding(text.lower())
            
            # Find best match among EOC examples
            best_similarity = 0.0
            best_example = None
            
            for example in self.eoc_examples:
                similarity = self._cosine_similarity(text_embedding, example.embedding)
                # Only consider high-confidence matches
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_example = example
            
            # Determine if it's EOC based on threshold
            is_eoc = best_similarity >= self.similarity_threshold
            confidence = best_similarity if is_eoc else 0.0
            example_id = best_example.id if best_example else None
            
            logger.debug(f"EOC detection: text='{text}', is_eoc={is_eoc}, confidence={confidence:.3f}, example={example_id}")
            
            return is_eoc, confidence, example_id
            
        except Exception as e:
            logger.error(f"Error in EOC detection: {e}")
            return False, 0.0, None
    
    def add_example(self, text: str, category: str = "general", confidence: float = 1.0) -> str:
        """Add a new EOC example"""
        import uuid
        example_id = f"eoc_{str(uuid.uuid4())[:8]}"
        
        embedding = self._generate_embedding(text)
        example = EOCExample(
            id=example_id,
            text=text,
            embedding=embedding,
            confidence=confidence,
            category=category
        )
        
        self.eoc_examples.append(example)
        logger.info(f"Added new EOC example: {example_id}")
        
        return example_id
    
    def get_examples(self) -> List[EOCExample]:
        """Get all EOC examples"""
        return self.eoc_examples[:]

# Global instance
eoc_detector = EOCEmbeddingDetector()

def detect_eoc_with_embedding(text: str) -> Tuple[bool, float, Optional[str]]:
    """
    Detect EOC using embedding similarity
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (is_eoc, confidence, example_id)
    """
    return eoc_detector.detect_eoc(text)

def add_eoc_example(text: str, category: str = "general", confidence: float = 1.0) -> str:
    """
    Add a new EOC example to the detector
    
    Args:
        text: Example EOC text
        category: Category of EOC (e.g., "thanks", "goodbye")
        confidence: Confidence level for this example
        
    Returns:
        ID of the added example
    """
    return eoc_detector.add_example(text, category, confidence)

def embed_text(text: str) -> List[float]:
    """
    Generate embedding for text
    
    Args:
        text: Input text
        
    Returns:
        Embedding vector
    """
    return eoc_detector._generate_embedding(text)