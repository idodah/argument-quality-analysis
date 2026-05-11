from datetime import date
from pydantic import BaseModel
from typing import Optional, Self

class ArgumentPair(BaseModel):
    thread_id: str
    topic: str
    original_post: str
    delta_argument: str
    nodelta_argument: str
    date: date | None
    summary: Optional[str] = None