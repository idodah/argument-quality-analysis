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

class ReferenceDocument(BaseModel):
    """An authoritative document refuting an antisemitic trope.

    Distinct from RagArgument: these carry no delta signal (nobody upvoted or
    was persuaded by them) but are factually authoritative, so the agent cites
    them for the historical record while CMV arguments supply persuasive form.
    `trope` names which documented myth the document addresses.
    """
    doc_id: str
    source: str          # "wikipedia" | "ushmm"
    trope: str
    title: str
    url: str
    text: str
    retrieved: date | None
