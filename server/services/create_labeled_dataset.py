"""
Script to create a labeled dataset from user feedback for EOC classifier retraining
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from ..database import db_manager
from .monitoring import get_metrics

logger = logging.getLogger(__name__)

async def collect_feedback_data() -> List[Dict[str, Any]]:
    """
    Collect feedback data from the database to create a labeled dataset.
    
    Returns:
        List of dictionaries containing feedback data with labels
    """
    try:
        # Get feedback data from conversations collection
        get_col = getattr(db_manager, 'get_collection', None)
        if not callable(get_col):
            logger.error("Database manager not available")
            return []
        
        convs = get_col('conversations')
        if convs is None:
            logger.error("Conversations collection not available")
            return []
        
        # Find conversations with feedback
        def _find_feedback_convs():
            return list(convs.find({
                "$or": [
                    {"feedback.type": {"$exists": True}},
                    {"feedback.useful": {"$exists": True}},
                    {"feedback.not_useful": {"$exists": True}}
                ]
            }))
        
        feedback_convs = await asyncio.to_thread(_find_feedback_convs)
        
        labeled_data = []
        for conv in feedback_convs:
            try:
                # Extract conversation data
                phone_number = conv.get('phone_number', '')
                transcript = conv.get('transcript', '')
                summary = conv.get('summary', '')
                eoc_confidence = conv.get('eoc_confidence', 0.0)
                eoc_detected_by = conv.get('eoc_detected_by', '')
                eoc_example_id = conv.get('eoc_example_id')
                
                # Extract feedback
                feedback = conv.get('feedback', {})
                feedback_type = feedback.get('type', '')
                feedback_text = feedback.get('text', '')
                feedback_timestamp = feedback.get('timestamp', '')
                
                # Determine label based on feedback
                # Positive feedback (useful) = True EOC detection
                # Negative feedback (not useful) = False EOC detection
                if feedback_type == 'useful' or feedback.get('useful', False):
                    label = 1  # True positive
                elif feedback_type == 'not_useful' or feedback.get('not_useful', False):
                    label = 0  # False positive
                else:
                    # Skip if feedback type is unclear
                    continue
                
                # Create labeled data entry
                data_entry = {
                    'phone_number': phone_number,
                    'transcript': transcript,
                    'summary': summary,
                    'eoc_confidence': eoc_confidence,
                    'eoc_detected_by': eoc_detected_by,
                    'eoc_example_id': str(eoc_example_id) if eoc_example_id else None,
                    'feedback_type': feedback_type,
                    'feedback_text': feedback_text,
                    'feedback_timestamp': feedback_timestamp,
                    'label': label,  # 1 for positive feedback (true EOC), 0 for negative feedback (false EOC)
                    'conversation_id': str(conv.get('_id', ''))
                }
                
                labeled_data.append(data_entry)
                
            except Exception as e:
                logger.warning(f"Error processing conversation {conv.get('_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Collected {len(labeled_data)} labeled feedback entries")
        return labeled_data
        
    except Exception as e:
        logger.error(f"Error collecting feedback data: {e}")
        return []

async def collect_false_positives_negatives() -> List[Dict[str, Any]]:
    """
    Collect false positives and negatives based on conversation patterns and timing.
    
    Returns:
        List of dictionaries containing potential false positives/negatives
    """
    try:
        get_col = getattr(db_manager, 'get_collection', None)
        if not callable(get_col):
            return []
        
        convs = get_col('conversations')
        if convs is None:
            return []
        
        # Find conversations that might be false positives/negatives
        def _find_suspicious_convs():
            return list(convs.find({
                "$or": [
                    # Conversations marked as EOC but with quick replies (false positive)
                    {
                        "state": "ended",
                        "messages_after_end": {"$gt": 0},
                        "time_to_reopen": {"$lt": 300}  # Less than 5 minutes
                    },
                    # Conversations not marked as EOC but with long gaps (false negative)
                    {
                        "state": "active",
                        "last_message_time": {"$lt": datetime.now().astimezone().timestamp() - 86400},  # 24 hours
                        "message_count": {"$gt": 5}
                    }
                ]
            }))
        
        suspicious_convs = await asyncio.to_thread(_find_suspicious_convs)
        
        labeled_data = []
        for conv in suspicious_convs:
            try:
                # Extract conversation data
                phone_number = conv.get('phone_number', '')
                transcript = conv.get('transcript', '')
                summary = conv.get('summary', '')
                eoc_confidence = conv.get('eoc_confidence', 0.0)
                eoc_detected_by = conv.get('eoc_detected_by', '')
                eoc_example_id = conv.get('eoc_example_id')
                state = conv.get('state', '')
                messages_after_end = conv.get('messages_after_end', 0)
                time_to_reopen = conv.get('time_to_reopen', 0)
                message_count = conv.get('message_count', 0)
                last_message_time = conv.get('last_message_time', 0)
                
                # Determine label based on pattern
                if state == "ended" and messages_after_end > 0 and time_to_reopen < 300:
                    # Likely false positive (EOC detected but conversation continued)
                    label = 0
                    label_type = "false_positive"
                elif state == "active" and message_count > 5 and last_message_time < (datetime.now().astimezone().timestamp() - 86400):
                    # Likely false negative (EOC not detected but conversation ended naturally)
                    label = 1
                    label_type = "false_negative"
                else:
                    continue
                
                # Create labeled data entry
                data_entry = {
                    'phone_number': phone_number,
                    'transcript': transcript,
                    'summary': summary,
                    'eoc_confidence': eoc_confidence,
                    'eoc_detected_by': eoc_detected_by,
                    'eoc_example_id': str(eoc_example_id) if eoc_example_id else None,
                    'state': state,
                    'messages_after_end': messages_after_end,
                    'time_to_reopen': time_to_reopen,
                    'message_count': message_count,
                    'last_message_time': last_message_time,
                    'label': label,
                    'label_type': label_type,
                    'conversation_id': str(conv.get('_id', ''))
                }
                
                labeled_data.append(data_entry)
                
            except Exception as e:
                logger.warning(f"Error processing suspicious conversation {conv.get('_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Collected {len(labeled_data)} suspicious conversation entries")
        return labeled_data
        
    except Exception as e:
        logger.error(f"Error collecting suspicious conversations: {e}")
        return []

async def create_labeled_dataset(output_file: str = None) -> str:
    """
    Create a complete labeled dataset combining feedback and suspicious patterns.
    
    Args:
        output_file: Optional path to save the dataset as JSON
        
    Returns:
        Path to the saved dataset file
    """
    try:
        # Collect all labeled data
        feedback_data = await collect_feedback_data()
        suspicious_data = await collect_false_positives_negatives()
        
        # Combine datasets
        all_labeled_data = feedback_data + suspicious_data
        
        # Add metadata
        dataset = {
            'created_at': datetime.now().astimezone().isoformat(),
            'total_entries': len(all_labeled_data),
            'feedback_entries': len(feedback_data),
            'suspicious_entries': len(suspicious_data),
            'data': all_labeled_data
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved labeled dataset to {output_file}")
            return output_file
        else:
            # Save to default location
            default_path = os.path.join(os.getcwd(), 'labeled_eoc_dataset.json')
            with open(default_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved labeled dataset to {default_path}")
            return default_path
            
    except Exception as e:
        logger.error(f"Error creating labeled dataset: {e}")
        raise

async def main():
    """Main function to run the dataset creation script"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting labeled dataset creation...")
        
        # Create the dataset
        dataset_path = await create_labeled_dataset()
        
        logger.info(f"Labeled dataset created successfully at {dataset_path}")
        
        # Print summary
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"\nDataset Summary:")
        print(f"  Total entries: {dataset['total_entries']}")
        print(f"  Feedback entries: {dataset['feedback_entries']}")
        print(f"  Suspicious entries: {dataset['suspicious_entries']}")
        print(f"  Saved to: {dataset_path}")
        
    except Exception as e:
        logger.error(f"Failed to create labeled dataset: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())