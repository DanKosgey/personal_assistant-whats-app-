-- Simplified schema for WhatsApp AI Agent
-- This migration creates a simplified users and conversations table structure

-- Enable UUID generation extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USERS TABLE - Simplified user profile storage
-- ============================================================================
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    phone VARCHAR(32) NOT NULL UNIQUE,          -- Canonical E.164 format
    name VARCHAR(100),                          -- User's display name
    address TEXT,                               -- User's address
    metadata JSONB,                             -- Arbitrary profile data
    consent BOOLEAN DEFAULT FALSE,              -- Memory storage consent
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_seen TIMESTAMP WITH TIME ZONE          -- Last interaction timestamp
);

-- Add indexes for performance
CREATE INDEX idx_users_phone ON users(phone);
CREATE INDEX idx_users_consent ON users(consent) WHERE consent = true;
CREATE INDEX idx_users_last_seen ON users(last_seen);

-- Add constraints
ALTER TABLE users 
ADD CONSTRAINT valid_phone CHECK (phone ~ '^\+?[1-9]\d{6,14}$');

-- ============================================================================
-- CONVERSATIONS TABLE - Simplified conversation storage
-- ============================================================================
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone VARCHAR(32) NOT NULL REFERENCES users(phone) ON DELETE CASCADE,
    message_id VARCHAR(255),                    -- WhatsApp message ID
    direction VARCHAR(10) NOT NULL,             -- 'in' or 'out'
    text TEXT NOT NULL,                         -- Message content
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Add indexes for performance
CREATE INDEX idx_conversations_phone ON conversations(phone);
CREATE INDEX idx_conversations_message_id ON conversations(message_id);
CREATE INDEX idx_conversations_direction ON conversations(direction);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);

-- Add constraints
ALTER TABLE conversations 
ADD CONSTRAINT valid_direction CHECK (direction IN ('in', 'out'));

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

-- Trigger for users table
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- ============================================================================

-- Function to get or create user
CREATE OR REPLACE FUNCTION get_or_create_user(p_phone VARCHAR(32), p_name VARCHAR(100) DEFAULT NULL)
RETURNS INTEGER AS $$
DECLARE
    user_id INTEGER;
BEGIN
    -- Try to get existing user
    SELECT id INTO user_id FROM users WHERE phone = p_phone;
    
    -- If user doesn't exist, create new one
    IF NOT FOUND THEN
        INSERT INTO users (phone, name, created_at, updated_at, last_seen)
        VALUES (p_phone, p_name, now(), now(), now())
        RETURNING id INTO user_id;
    END IF;
    
    -- Update last_seen timestamp
    UPDATE users SET last_seen = now() WHERE id = user_id;
    
    RETURN user_id;
END;
$$ LANGUAGE plpgsql;

-- Function to save user name
CREATE OR REPLACE FUNCTION save_user(p_phone VARCHAR(32), p_name VARCHAR(100))
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE users 
    SET name = p_name, updated_at = now()
    WHERE phone = p_phone;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to request consent
CREATE OR REPLACE FUNCTION request_consent(p_phone VARCHAR(32))
RETURNS BOOLEAN AS $$
BEGIN
    -- This would typically trigger a message to the user
    -- For now, we'll just return true to indicate the function exists
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to get user
CREATE OR REPLACE FUNCTION get_user(p_phone VARCHAR(32))
RETURNS TABLE(id INTEGER, phone VARCHAR(32), name VARCHAR(100), address TEXT, metadata JSONB, consent BOOLEAN, created_at TIMESTAMP WITH TIME ZONE, updated_at TIMESTAMP WITH TIME ZONE, last_seen TIMESTAMP WITH TIME ZONE) AS $$
BEGIN
    RETURN QUERY
    SELECT u.id, u.phone, u.name, u.address, u.metadata, u.consent, u.created_at, u.updated_at, u.last_seen
    FROM users u
    WHERE u.phone = p_phone;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Add comments for documentation
COMMENT ON TABLE users IS 'Simplified user profiles with basic information';
COMMENT ON TABLE conversations IS 'Conversation messages with direction and content';
COMMENT ON COLUMN users.phone IS 'Canonical E.164 format phone number';
COMMENT ON COLUMN users.metadata IS 'Arbitrary profile data in JSON format';
COMMENT ON COLUMN conversations.direction IS 'Message direction: in (from user) or out (to user)';