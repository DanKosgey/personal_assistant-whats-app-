-- Migration to add EOC (End of Conversation) detection columns to profile summaries table
-- These columns will store confidence scores and detection methods for conversation summaries

-- Add columns for EOC detection metadata
ALTER TABLE profile_summaries 
ADD COLUMN eoc_confidence DECIMAL(3,2) CHECK (eoc_confidence >= 0.0 AND eoc_confidence <= 1.0),
ADD COLUMN eoc_detected_by VARCHAR(64),
ADD COLUMN eoc_example_id UUID;

-- Create indexes for the new columns
CREATE INDEX idx_profile_summaries_eoc_confidence ON profile_summaries(eoc_confidence) WHERE eoc_confidence IS NOT NULL;
CREATE INDEX idx_profile_summaries_eoc_detected_by ON profile_summaries(eoc_detected_by) WHERE eoc_detected_by IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN profile_summaries.eoc_confidence IS 'EOC detection confidence score (0.0-1.0)';
COMMENT ON COLUMN profile_summaries.eoc_detected_by IS 'Method used for EOC detection (keyword_matching, embedding_similarity, classifier)';
COMMENT ON COLUMN profile_summaries.eoc_example_id IS 'Reference to the EOC example that triggered detection (for embedding-based detection)';