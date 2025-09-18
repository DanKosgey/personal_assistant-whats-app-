"""
EOC (End of Conversation) Classifier
Implements a simple classifier for EOC probability scoring
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EOCFeatures:
    """Features used for EOC classification"""
    keyword_score: float = 0.0
    embedding_score: float = 0.0
    message_length: float = 0.0
    punctuation_count: float = 0.0
    question_mark_count: float = 0.0
    exclamation_mark_count: float = 0.0
    contains_thanks: bool = False
    contains_goodbye: bool = False
    contains_confirmation: bool = False

class EOCClassifier:
    """Simple classifier for EOC probability scoring"""
    
    def __init__(self):
        # Simple weights for features (in a real implementation, these would be learned)
        self.weights = {
            'keyword_score': 0.3,
            'embedding_score': 0.4,
            'message_length': -0.1,  # Shorter messages more likely to be EOC
            'punctuation_count': 0.05,
            'question_mark_count': -0.2,  # Questions less likely to be EOC
            'exclamation_mark_count': 0.1,
            'contains_thanks': 0.5,
            'contains_goodbye': 0.6,
            'contains_confirmation': 0.4
        }
        
        # Threshold for classification
        self.threshold = 0.5
        
        logger.info("EOC classifier initialized")
    
    def extract_features(self, text: str, keyword_score: float = 0.0, embedding_score: float = 0.0) -> EOCFeatures:
        """Extract features from text for EOC classification"""
        features = EOCFeatures()
        
        # Normalize text
        lower_text = text.lower().strip()
        
        # Basic scores
        features.keyword_score = keyword_score
        features.embedding_score = embedding_score
        
        # Text statistics
        features.message_length = len(lower_text) / 100.0  # Normalize by 100 chars
        features.punctuation_count = len(re.findall(r'[.!?;]', lower_text)) / 10.0  # Normalize
        features.question_mark_count = lower_text.count('?') / 5.0  # Normalize
        features.exclamation_mark_count = lower_text.count('!') / 5.0  # Normalize
        
        # Keyword presence
        thanks_keywords = ['thank', 'thanks', 'thx', 'appreciate']
        goodbye_keywords = ['bye', 'goodbye', 'see you', 'farewell', 'later']
        confirmation_keywords = ['ok', 'okay', 'yes', 'yeah', 'sure', 'sounds good', 'perfect', 'great']
        
        features.contains_thanks = any(keyword in lower_text for keyword in thanks_keywords)
        features.contains_goodbye = any(keyword in lower_text for keyword in goodbye_keywords)
        features.contains_confirmation = any(keyword in lower_text for keyword in confirmation_keywords)
        
        return features
    
    def calculate_probability(self, features: EOCFeatures) -> float:
        """Calculate EOC probability based on features"""
        # Weighted sum of features
        score = 0.0
        for feature_name, weight in self.weights.items():
            feature_value = getattr(features, feature_name, 0.0)
            score += weight * feature_value
        
        # Apply sigmoid to get probability (0-1 range)
        import math
        probability = 1.0 / (1.0 + math.exp(-score))
        
        return probability
    
    def classify(self, text: str, keyword_score: float = 0.0, embedding_score: float = 0.0) -> Tuple[bool, float]:
        """
        Classify text as EOC or not with probability score
        
        Returns:
            Tuple of (is_eoc, probability)
        """
        try:
            # Extract features
            features = self.extract_features(text, keyword_score, embedding_score)
            
            # Calculate probability
            probability = self.calculate_probability(features)
            
            # Classify based on threshold
            is_eoc = probability >= self.threshold
            
            logger.debug(f"EOC classification: text='{text[:50]}...', is_eoc={is_eoc}, probability={probability:.3f}")
            
            return is_eoc, probability
            
        except Exception as e:
            logger.error(f"Error in EOC classification: {e}")
            return False, 0.0

# Global instance
eoc_classifier = EOCClassifier()

def classify_eoc(text: str, keyword_score: float = 0.0, embedding_score: float = 0.0) -> Tuple[bool, float]:
    """
    Classify text as EOC with probability score
    
    Args:
        text: Input text to classify
        keyword_score: Keyword-based EOC score (0-1)
        embedding_score: Embedding-based EOC score (0-1)
        
    Returns:
        Tuple of (is_eoc, probability)
    """
    return eoc_classifier.classify(text, keyword_score, embedding_score)