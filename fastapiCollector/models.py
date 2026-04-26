# models.py — SQLAlchemy ORM models for Sentinel

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)          # UUID or provided ID
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=True)   # plain text for prototype
    source = Column(String, default="booth")        # "booth" for hackathon
    created_at = Column(DateTime, server_default=func.now())


class RawEnrollment(Base):
    __tablename__ = "raw_enrollment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    session_id = Column(String, nullable=False)     # groups one full enrollment
    keystrokes_json = Column(Text)                  # JSON array of KeystrokeEvent
    scrolls_json = Column(Text)                     # JSON array of ScrollEvent
    imu_json = Column(Text)                         # JSON array of IMUSample
    collected_at = Column(DateTime, server_default=func.now())


class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    stats_vector = Column(Text)                     # JSON array of 13 floats
    gaussian_profile = Column(Text, nullable=True)  # JSON: Gaussian biometric profile (mean + std)
    enrolled_at = Column(DateTime, server_default=func.now())