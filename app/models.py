from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text,LargeBinary,JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    messages = relationship("Message", back_populates="user")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(JSON)
    direction = Column(String)  # "incoming" or "outgoing"
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="messages")



class DeviceDocument(Base):
    __tablename__ = "device_documents"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, nullable=False, index=True)
    device_pdf = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())