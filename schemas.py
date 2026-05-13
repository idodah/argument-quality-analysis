from datetime import date
from pydantic import BaseModel

class ArgumentPair(BaseModel):
    thread_id: str
    topic: str
    original_post: str
    delta_argument: str
    nodelta_argument: str
    date: date | None