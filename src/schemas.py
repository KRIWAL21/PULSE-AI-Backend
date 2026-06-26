"""
Pydantic schemas for request/response validation and serialization.
"""

import re
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, field_validator


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ──────────────────────────────────────────────────────────────────────────
# AUTH SCHEMAS
# ──────────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    """Request schema for user registration."""
    email: str
    password: str
    full_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123",
                "full_name": "John Doe"
            }
        }

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Validate email format without requiring email-validator."""
        normalized = value.strip().lower()
        if not EMAIL_PATTERN.match(normalized):
            raise ValueError("Invalid email address")
        return normalized


class UserLogin(BaseModel):
    """Request schema for user login."""
    email: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Validate email format without requiring email-validator."""
        normalized = value.strip().lower()
        if not EMAIL_PATTERN.match(normalized):
            raise ValueError("Invalid email address")
        return normalized


class TokenResponse(BaseModel):
    """Response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    """User data returned to client (no sensitive fields)."""
    id: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────────────────────────────────
# CHAT SCHEMAS
# ──────────────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    """Request body for chat message."""
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "What is hypertension?"}
        }


class AskRequest(BaseModel):
    """Request for RAG query with history."""
    question: str
    chat_history: List[dict] = []

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the side effects?",
                "chat_history": [
                    {"role": "user", "content": "Tell me about aspirin"},
                    {"role": "assistant", "content": "Aspirin is..."}
                ]
            }
        }


class AskResponse(BaseModel):
    """Response from RAG endpoint."""
    answer: str
    sources: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The main side effects are...",
                "sources": ["medical_guide.pdf"]
            }
        }


class MessageResponse(BaseModel):
    """Message data returned to client."""
    id: str
    conversation_id: str
    sender: str  # "user" or "bot"
    text: str
    sources: Optional[List[str]] = None
    created_at: datetime

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, value):
        if isinstance(value, str):
            import json
            try:
                return json.loads(value)
            except Exception:
                return []
        return value

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    title: Optional[str] = "New Chat"

    class Config:
        json_schema_extra = {
            "example": {"title": "Medical Questions"}
        }


class ConversationUpdate(BaseModel):
    """Request to update conversation title."""
    title: str

    class Config:
        json_schema_extra = {
            "example": {"title": "Aspirin Discussion"}
        }


class ConversationResponse(BaseModel):
    """Conversation data returned to client."""
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None

    class Config:
        from_attributes = True


class SummarizeRequest(BaseModel):
    """Request to summarize a conversation."""
    messages: List[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"sender": "user", "text": "What is diabetes?"},
                    {"sender": "bot", "text": "Diabetes is..."}
                ]
            }
        }


class SummarizeResponse(BaseModel):
    """Summary response."""
    summary: str

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "## Topics Discussed\n- Diabetes types\n- Symptoms"
            }
        }
