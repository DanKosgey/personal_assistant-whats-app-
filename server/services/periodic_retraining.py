"""
Periodic retraining system for EOC classifier
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import json

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .create_labeled_dataset import create_labeled_dataset
from .train_eoc_classifier import EOCTrainer

logger = logging.getLogger(__name__)

class PeriodicRetrainer:
    """System for periodically retraining the EOC classifier"""
    
    def __init__(self, retraining_interval_hours: int = 24):
        self.retraining_interval = timedelta(hours=retraining_interval_hours)
        self.last_training_time: Optional[datetime] = None
        self.is_training = False
        
    async def should_retrain(self) -> bool:
        """
        Check if classifier should be retrained based on time interval and data availability.
        
        Returns:
            True if retraining should occur, False otherwise
        """
        # Don't retrain if already training
        if self.is_training:
            return False
            
        # Always retrain if never trained before
        if self.last_training_time is None:
            return True
            
        # Check if enough time has passed
        time_since_last = datetime.utcnow() - self.last_training_time
        return time_since_last >= self.retraining_interval
    
    async def check_data_sufficiency(self, min_samples: int = 50) -> bool:
        """
        Check if there's sufficient data for retraining.
        
        Args:
            min_samples: Minimum number of samples required for training
            
        Returns:
            True if sufficient data available, False otherwise
        """
        try:
            # Create a temporary dataset to check size
            dataset_path = await create_labeled_dataset()
            
            # Load and check dataset size
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            total_entries = dataset.get('total_entries', 0)
            os.remove(dataset_path)  # Clean up temporary file
            
            sufficient = total_entries >= min_samples
            logger.info(f"Data sufficiency check: {total_entries} entries, {'sufficient' if sufficient else 'insufficient'}")
            
            return sufficient
            
        except Exception as e:
            logger.error(f"Error checking data sufficiency: {e}")
            return False
    
    async def retrain_classifier(self) -> bool:
        """
        Perform classifier retraining if conditions are met.
        
        Returns:
            True if retraining was successful, False otherwise
        """
        try:
            # Check if retraining is needed
            if not await self.should_retrain():
                logger.info("Retraining not needed at this time")
                return False
            
            # Check if there's sufficient data
            if not await self.check_data_sufficiency():
                logger.info("Insufficient data for retraining")
                return False
            
            logger.info("Starting classifier retraining...")
            self.is_training = True
            
            # Create labeled dataset
            dataset_path = await create_labeled_dataset()
            logger.info(f"Created dataset at {dataset_path}")
            
            # Initialize trainer
            trainer = EOCTrainer()
            
            # Train classifier
            results = trainer.train_classifier(dataset_path)
            logger.info("Classifier training completed")
            
            # Save model
            model_path = trainer.save_model()
            logger.info(f"Model saved to {model_path}")
            
            # Save results
            results_path = os.path.join(os.getcwd(), 'training_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Training results saved to {results_path}")
            
            # Update last training time
            self.last_training_time = datetime.utcnow()
            self.is_training = False
            
            logger.info("Classifier retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during classifier retraining: {e}")
            self.is_training = False
            return False
    
    async def run_periodic_retraining(self, check_interval_minutes: int = 60):
        """
        Run periodic retraining in a loop.
        
        Args:
            check_interval_minutes: How often to check if retraining is needed
        """
        logger.info(f"Starting periodic retraining system (check interval: {check_interval_minutes} minutes)")
        
        while True:
            try:
                # Check if retraining is needed
                if await self.should_retrain():
                    logger.info("Retraining conditions met, checking data sufficiency...")
                    
                    # Attempt retraining
                    success = await self.retrain_classifier()
                    if success:
                        logger.info("Periodic retraining completed successfully")
                    else:
                        logger.info("Periodic retraining skipped or failed")
                else:
                    logger.debug("Retraining not needed at this time")
                
                # Wait before next check
                await asyncio.sleep(check_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("Periodic retraining cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic retraining loop: {e}")
                # Wait before retrying
                await asyncio.sleep(check_interval_minutes * 60)

# Global retraining instance
retrainer = PeriodicRetrainer()

async def start_periodic_retraining(check_interval_minutes: int = 60):
    """
    Start the periodic retraining system.
    
    Args:
        check_interval_minutes: How often to check if retraining is needed
    """
    await retrainer.run_periodic_retraining(check_interval_minutes)

async def trigger_manual_retraining() -> bool:
    """
    Manually trigger classifier retraining.
    
    Returns:
        True if retraining was successful, False otherwise
    """
    logger.info("Manual retraining triggered")
    return await retrainer.retrain_classifier()

async def get_retraining_status() -> Dict[str, Any]:
    """
    Get the current status of the retraining system.
    
    Returns:
        Dictionary with retraining status information
    """
    return {
        'is_training': retrainer.is_training,
        'last_training_time': retrainer.last_training_time.isoformat() if retrainer.last_training_time else None,
        'next_training_due': (
            (retrainer.last_training_time + retrainer.retraining_interval).isoformat()
            if retrainer.last_training_time else "As soon as sufficient data is available"
        ),
        'retraining_interval_hours': retrainer.retraining_interval.total_seconds() / 3600
    }

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run manual retraining for testing
    asyncio.run(trigger_manual_retraining())