-- WhatsApp Profile Management System Database Schema
-- Production-ready schema with indexing, constraints, and audit logging
-- Based on the comprehensive record-keeper design specification

-- Enable UUID generation extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- PROFILES TABLE - Main user profile storage with versioning
-- ============================================================================
CREATE TABLE profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone VARCHAR(32) NOT NULL UNIQUE,           -- Primary identity (E.164 format)
    name TEXT,                                   -- User's display name
    display_name TEXT,                           -- What the bot uses to address user
    persona VARCHAR(64),                         -- User archetype (supportive_customer, trader, developer)
    description TEXT,                            -- AI-generated summary of user
    language VARCHAR(8) DEFAULT 'en',            -- ISO 639-1 language code
    timezone VARCHAR(64),                        -- IANA timezone identifier
    consent BOOLEAN DEFAULT FALSE,               -- Has user opted into memory storage
    consent_date TIMESTAMP WITH TIME ZONE,      -- When consent was given/revoked
    last_seen TIMESTAMP WITH TIME ZONE,         -- Last interaction timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    version INTEGER DEFAULT 1,                  -- Optimistic locking version
    attributes JSONB DEFAULT '{}',              -- Flexible key/value for custom facts
    tags TEXT[] DEFAULT '{}',                   -- Array of tags for quick filtering
    updated_by TEXT DEFAULT 'system',           -- Who/what updated (bot, admin, user)
    status VARCHAR(16) DEFAULT 'active',        -- active, inactive, merged, deleted
    merged_into UUID REFERENCES profiles(id),   -- If merged, points to primary profile
    deleted_at TIMESTAMP WITH TIME ZONE,        -- Soft delete timestamp
    
    -- Constraints
    CONSTRAINT valid_phone CHECK (phone ~ '^\+?[1-9]\d{6,14}$'),
    CONSTRAINT valid_language CHECK (language ~ '^[a-z]{2}(-[A-Z]{2})?$'),
    CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'merged', 'deleted')),
    CONSTRAINT valid_persona CHECK (persona IS NULL OR length(persona) <= 64)
);

-- ============================================================================
-- PROFILE_HISTORY TABLE - Immutable audit log of all changes
-- ============================================================================
CREATE TABLE profile_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    changed_by TEXT NOT NULL,                    -- Actor who made the change
    change_type VARCHAR(16) NOT NULL,            -- create, update, delete, merge, consent
    change_data JSONB NOT NULL,                  -- Full diff or snapshot of change
    reason TEXT,                                 -- Human-readable reason for change
    session_id TEXT,                             -- Optional session/conversation tracking
    ip_address INET,                             -- For security auditing
    user_agent TEXT,                             -- Request source information
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Constraints
    CONSTRAINT valid_change_type CHECK (change_type IN ('create', 'update', 'delete', 'merge', 'consent', 'revert'))
);

-- ============================================================================
-- PROFILE_TAGS TABLE - Normalized tags for efficient filtering
-- ============================================================================
CREATE TABLE profile_tags (
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_by TEXT DEFAULT 'system',
    
    PRIMARY KEY (profile_id, tag),
    
    -- Constraints
    CONSTRAINT valid_tag CHECK (length(tag) > 0 AND length(tag) <= 64)
);

-- ============================================================================
-- PROFILE_SUMMARIES TABLE - AI-generated summaries with embeddings
-- ============================================================================
CREATE TABLE profile_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,                       -- AI-generated summary
    summary_type VARCHAR(32) DEFAULT 'conversation', -- conversation, personality, preferences
    confidence_score DECIMAL(3,2),              -- AI confidence in summary (0.0-1.0)
    source_messages TEXT[],                      -- Reference to source message IDs
    embedding VECTOR(1536),                     -- OpenAI embedding (requires pgvector)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_by TEXT DEFAULT 'ai_summarizer',
    expires_at TIMESTAMP WITH TIME ZONE,        -- Optional expiry for temporary summaries
    
    -- Constraints
    CONSTRAINT valid_confidence CHECK (confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)),
    CONSTRAINT valid_summary_type CHECK (summary_type IN ('conversation', 'personality', 'preferences', 'behavior', 'interests'))
);

-- ============================================================================
-- PROFILE_RELATIONSHIPS TABLE - For handling connections between profiles
-- ============================================================================
CREATE TABLE profile_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    primary_profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    related_profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    relationship_type VARCHAR(32) NOT NULL,     -- family, colleague, duplicate, alias
    strength DECIMAL(3,2) DEFAULT 0.5,          -- Relationship strength (0.0-1.0)
    notes TEXT,                                  -- Additional context
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_by TEXT DEFAULT 'system',
    
    -- Constraints
    CONSTRAINT no_self_reference CHECK (primary_profile_id != related_profile_id),
    CONSTRAINT valid_relationship_type CHECK (relationship_type IN ('family', 'colleague', 'duplicate', 'alias', 'business', 'friend')),
    CONSTRAINT valid_strength CHECK (strength >= 0.0 AND strength <= 1.0),
    
    -- Unique constraint to prevent duplicate relationships
    UNIQUE(primary_profile_id, related_profile_id, relationship_type)
);

-- ============================================================================
-- PROFILE_CONSENTS TABLE - Detailed consent tracking for GDPR compliance
-- ============================================================================
CREATE TABLE profile_consents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    consent_type VARCHAR(32) NOT NULL,          -- memory_storage, data_processing, marketing
    granted BOOLEAN NOT NULL,                   -- True if consent granted, false if revoked
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE,        -- Optional consent expiry
    consent_method VARCHAR(32),                 -- whatsapp_message, api_call, web_form
    ip_address INET,                            -- For legal compliance
    user_agent TEXT,                            -- Request source
    legal_basis VARCHAR(64),                    -- GDPR legal basis
    notes TEXT,                                 -- Additional context
    
    -- Constraints
    CONSTRAINT valid_consent_type CHECK (consent_type IN ('memory_storage', 'data_processing', 'marketing', 'analytics', 'sharing')),
    CONSTRAINT valid_consent_method CHECK (consent_method IN ('whatsapp_message', 'api_call', 'web_form', 'phone_call', 'email'))
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX idx_profiles_phone ON profiles(phone) WHERE status = 'active';
CREATE INDEX idx_profiles_last_seen ON profiles(last_seen) WHERE status = 'active';
CREATE INDEX idx_profiles_status ON profiles(status);
CREATE INDEX idx_profiles_persona ON profiles(persona) WHERE persona IS NOT NULL;
CREATE INDEX idx_profiles_consent ON profiles(consent) WHERE consent = true;
CREATE INDEX idx_profiles_updated_at ON profiles(updated_at);

-- GIN indexes for JSONB and array fields
CREATE INDEX idx_profiles_attributes ON profiles USING GIN(attributes);
CREATE INDEX idx_profiles_tags ON profiles USING GIN(tags);

-- History table indexes
CREATE INDEX idx_profile_history_profile_id ON profile_history(profile_id);
CREATE INDEX idx_profile_history_created_at ON profile_history(created_at);
CREATE INDEX idx_profile_history_change_type ON profile_history(change_type);
CREATE INDEX idx_profile_history_changed_by ON profile_history(changed_by);

-- Profile summaries indexes
CREATE INDEX idx_profile_summaries_profile_id ON profile_summaries(profile_id);
CREATE INDEX idx_profile_summaries_type ON profile_summaries(summary_type);
CREATE INDEX idx_profile_summaries_created_at ON profile_summaries(created_at);
CREATE INDEX idx_profile_summaries_confidence ON profile_summaries(confidence_score) WHERE confidence_score IS NOT NULL;

-- Relationships indexes
CREATE INDEX idx_profile_relationships_primary ON profile_relationships(primary_profile_id);
CREATE INDEX idx_profile_relationships_related ON profile_relationships(related_profile_id);
CREATE INDEX idx_profile_relationships_type ON profile_relationships(relationship_type);

-- Consents indexes
CREATE INDEX idx_profile_consents_profile_id ON profile_consents(profile_id);
CREATE INDEX idx_profile_consents_type ON profile_consents(consent_type);
CREATE INDEX idx_profile_consents_granted ON profile_consents(granted);
CREATE INDEX idx_profile_consents_granted_at ON profile_consents(granted_at);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for profiles table
CREATE TRIGGER update_profiles_updated_at 
    BEFORE UPDATE ON profiles 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- ============================================================================

-- Function to safely increment version with optimistic locking
CREATE OR REPLACE FUNCTION increment_profile_version(profile_uuid UUID, expected_version INTEGER)
RETURNS INTEGER AS $$
DECLARE
    current_version INTEGER;
    new_version INTEGER;
BEGIN
    -- Get current version
    SELECT version INTO current_version FROM profiles WHERE id = profile_uuid;
    
    -- Check if version matches expected
    IF current_version != expected_version THEN
        RAISE EXCEPTION 'Version conflict: expected %, got %', expected_version, current_version;
    END IF;
    
    -- Increment version
    new_version := current_version + 1;
    UPDATE profiles SET version = new_version WHERE id = profile_uuid;
    
    RETURN new_version;
END;
$$ LANGUAGE plpgsql;

-- Function to create audit log entry
CREATE OR REPLACE FUNCTION create_audit_entry(
    p_profile_id UUID,
    p_changed_by TEXT,
    p_change_type TEXT,
    p_change_data JSONB,
    p_reason TEXT DEFAULT NULL,
    p_session_id TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    audit_id UUID;
BEGIN
    INSERT INTO profile_history (
        profile_id, changed_by, change_type, change_data, reason, session_id
    ) VALUES (
        p_profile_id, p_changed_by, p_change_type, p_change_data, p_reason, p_session_id
    ) RETURNING id INTO audit_id;
    
    RETURN audit_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check and update consent
CREATE OR REPLACE FUNCTION update_consent(
    p_profile_id UUID,
    p_consent_type TEXT,
    p_granted BOOLEAN,
    p_method TEXT DEFAULT 'api_call'
)
RETURNS UUID AS $$
DECLARE
    consent_id UUID;
BEGIN
    -- Insert new consent record
    INSERT INTO profile_consents (
        profile_id, consent_type, granted, consent_method
    ) VALUES (
        p_profile_id, p_consent_type, p_granted, p_method
    ) RETURNING id INTO consent_id;
    
    -- Update main profile consent flag if it's memory storage consent
    IF p_consent_type = 'memory_storage' THEN
        UPDATE profiles 
        SET consent = p_granted, consent_date = now() 
        WHERE id = p_profile_id;
    END IF;
    
    RETURN consent_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active profiles with latest consent info
CREATE VIEW active_profiles AS
SELECT 
    p.*,
    pc.granted as memory_consent,
    pc.granted_at as memory_consent_date
FROM profiles p
LEFT JOIN LATERAL (
    SELECT granted, granted_at
    FROM profile_consents
    WHERE profile_id = p.id AND consent_type = 'memory_storage'
    ORDER BY granted_at DESC
    LIMIT 1
) pc ON true
WHERE p.status = 'active' AND p.deleted_at IS NULL;

-- Profile summary with latest AI-generated description
CREATE VIEW profile_summaries_latest AS
SELECT DISTINCT ON (profile_id)
    profile_id,
    summary,
    summary_type,
    confidence_score,
    created_at
FROM profile_summaries
WHERE expires_at IS NULL OR expires_at > now()
ORDER BY profile_id, created_at DESC;

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default system user for tracking automated changes
-- Using a valid phone number format that passes the constraint
INSERT INTO profiles (
    id, phone, name, display_name, persona, description, 
    consent, status, updated_by
) VALUES (
    uuid_generate_v4(), 
    '+10000000000', 
    'System', 
    'AI Assistant', 
    'system', 
    'Internal system profile for automated operations',
    true, 
    'active', 
    'migration'
) ON CONFLICT (phone) DO NOTHING;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO whatsapp_bot_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO whatsapp_bot_user;

-- Create composite indexes for common query patterns
CREATE INDEX idx_profiles_composite_lookup ON profiles(phone, status, consent) WHERE status = 'active';
CREATE INDEX idx_profiles_persona_consent ON profiles(persona, consent) WHERE status = 'active' AND consent = true;

COMMENT ON TABLE profiles IS 'Main user profiles with versioning and soft delete support';
COMMENT ON TABLE profile_history IS 'Immutable audit log of all profile changes';
COMMENT ON TABLE profile_tags IS 'Normalized tags for efficient profile categorization';
COMMENT ON TABLE profile_summaries IS 'AI-generated summaries with optional embeddings';
COMMENT ON TABLE profile_relationships IS 'Relationships and connections between profiles';
COMMENT ON TABLE profile_consents IS 'Detailed consent tracking for GDPR compliance';