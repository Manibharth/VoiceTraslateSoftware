from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False)
    filename = Column(String(256))
    status = Column(String(32), default="pending")  # pending, processing, done, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    output_path = Column(String(512))
    subtitle_path = Column(String(512))
    error_message = Column(Text)

    speakers = relationship("Speaker", back_populates="job", cascade="all, delete-orphan")
    segments = relationship("Segment", back_populates="job", cascade="all, delete-orphan")


class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), ForeignKey("jobs.job_id"))
    speaker_label = Column(String(32))  # SPEAKER_00, SPEAKER_01...
    gender = Column(String(16))         # Male / Female
    age_type = Column(String(16))       # Adult / Child
    tone = Column(String(16))           # Happy / Angry / Sad / Neutral
    pitch_mean = Column(Float)
    pitch_std = Column(Float)
    speech_rate = Column(Float)
    voice_profile = Column(String(64))  # Hero Male, Heroine Female, Child Voice, Comedian Voice
    voice_model = Column(String(128))   # TTS model identifier
    override_voice = Column(String(64)) # user-selected override

    job = relationship("Job", back_populates="speakers")


class Segment(Base):
    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), ForeignKey("jobs.job_id"))
    speaker_label = Column(String(32))
    start_time = Column(Float)
    end_time = Column(Float)
    original_text = Column(Text)
    translated_text = Column(Text)
    audio_path = Column(String(512))
    language = Column(String(16), default="en")

    job = relationship("Job", back_populates="segments")


def get_engine(db_path="database/voice_translate.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine
