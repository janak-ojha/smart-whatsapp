# Pydantic schemas
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class MessageBase(BaseModel):
    content: str
    direction: str

class MessageCreate(MessageBase):
    pass

class Message(MessageBase):
    id: int
    user_id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

class UserBase(BaseModel):
    phone: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    messages: List[Message] = []
    
    class Config:
        from_attributes = True

# WhatsApp webhook schemas
class WhatsAppTextMessage(BaseModel):
    body: str

class WhatsAppMessage(BaseModel):
    from_: str
    text: Optional[WhatsAppTextMessage]
    
    class Config:
        fields = {
            'from_': 'from'
        }

class WhatsAppWebhook(BaseModel):
    messages: List[WhatsAppMessage]