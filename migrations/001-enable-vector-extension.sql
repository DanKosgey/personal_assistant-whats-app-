-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is enabled
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';