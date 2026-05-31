"""Pydantic record types shared across the preprocessing and RAG pipelines."""

from datetime import date
from pydantic import BaseModel


class ArgumentPair(BaseModel):
    thread_id: str
    topic: str
    original_post: str
    delta_argument: str
    nodelta_argument: str
    date: date | None


class RagArgument(BaseModel):
    """A single delta-awarded argument with its post context, for RAG retrieval."""
    thread_id: str
    comment_id: str
    topic: str
    original_post: str
    argument: str
    score: int
    date: date | None