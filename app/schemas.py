# Pydantic schemas
from pydantic import BaseModel,Field
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
    from_: str = Field(alias="from")
    text: Optional[WhatsAppTextMessage]
class WhatsAppWebhook(BaseModel):
    messages: List[WhatsAppMessage]

# for adding document
class DeviceDocumentResponse(BaseModel):
    id: int
    device_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class QuestionRequest(BaseModel):
    question: str
    device_id: int        