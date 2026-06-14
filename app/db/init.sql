-- Run this once against your Azure PostgreSQL database before starting the app.
-- Only enables the pgvector extension.
-- Tables and the tsvector trigger are created automatically by the app on startup.

CREATE EXTENSION IF NOT EXISTS vector;
