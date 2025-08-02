from pydantic import BaseModel

class InvokeModel(BaseModel):
    content: str
    session_id: str = "default_session"  # Default session ID if not provided