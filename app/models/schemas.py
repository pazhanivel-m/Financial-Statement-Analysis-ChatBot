"""
Pydantic v2 request / response schemas for all API endpoints.
Keeping schemas separate from ORM models avoids tight coupling and
makes API contracts explicit.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RetrievalMethod(StrEnum):
    SEMANTIC = "semantic"
    BM25_POSTGRES = "bm25_postgres"


# ── Document schemas ──────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    company: str
    total_pages: int
    total_chunks: int
    message: str


class DocumentListItem(BaseModel):
    document_id: uuid.UUID
    filename: str
    company: str
    total_pages: int | None
    fiscal_year: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Chat schemas ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's financial question.",
        examples=["Compare the EBITDA margin of TCS and Infosys for FY2024."],
    )
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn context.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class ToolCall(BaseModel):
    """Records which tool the LLM invoked and with what parameters."""
    tool_name: str
    companies: list[str]
    parameters: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    query: str
    tools_used: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    yahoo_finance_loaded: bool
