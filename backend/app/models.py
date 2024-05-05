# Path: app/models.py
from uuid import uuid4

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """
    Base model for all database entities.
    Provides common attributes like ID, creation, and update timestamps.
    Inherits from SQLAlchemy's DeclarativeBase.
    """

    __abstract__ = True

    id = Column(UUID, primary_key=True, default=uuid4)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# Data Orchestration models


class OrchestrationEvent(Base):
    """
    Model for orchestration events.
    Contains orchestration pipeline event status, message, and reference to the pipeline.
    Useful for tracking the status of ETL jobs.
    """

    __tablename__ = "orchestration_events"
    status = Column(String, default="pending")  # running, success, failure
    message = Column(Text)
    payload = Column(JSON)
    environment = Column(String)
    source_uri = Column(JSON)
    destination_uri = Column(JSON)
    pipeline_id = Column(UUID, ForeignKey("orchestration_pipelines.id"))
    orchestration_pipeline = relationship(
        "OrchestrationPipeline", back_populates="orchestration_events"
    )


class OrchestrationPipeline(Base):
    """
    Model for ETL (Extract, Transform, Load) pipelines.
    Contains name, description, source, destination, and parameters.
    Linked to orchestration events via one-to-many relationship.
    """

    __tablename__ = "orchestration_pipelines"
    name = Column(String, index=True, nullable=False)
    description = Column(Text)
    definition = Column(JSON)
    user_id = Column(UUID, ForeignKey("users.id"))
    user = relationship("User", back_populates="orchestration_pipelines")
    orchestration_events = relationship(
        "OrchestrationEvent", back_populates="orchestration_pipeline"
    )


# Extractor models
class ExtractorExample(Base):
    """A representation of an example.

    Examples consist of content together with the expected output.

    The output is a JSON object that is expected to be extracted from the content.

    The JSON object should be valid according to the schema of the associated extractor.

    The JSON object is defined by the schema of the associated extractor, so
    it's perfectly fine for a given example to represent the extraction
    of multiple instances of some object from the content since
    the JSON schema can represent a list of objects.
    """

    __tablename__ = "extractor_examples"
    content = Column(Text, nullable=False, comment="The input portion of the example.")
    output = Column(JSONB, comment="The output associated with the example.")
    extractor_id = Column(UUID, ForeignKey("extractors.id"))
    extractor = relationship("Extractor", back_populates="extractor_examples")

    def __repr__(self) -> str:
        return f"<ExtractorExample(uuid={self.id}, content={self.content[:20]}>"


class Extractor(Base):
    """
    Represents an extractor for parsing structured data from unstructured text.
    Contains name, description, schema, instruction, etc.
    Linked to users and examples via many-to-one and one-to-many relationships.
    """
    __tablename__ = "extractors"
    name = Column(String, index=True, nullable=False)
    description = Column(Text)
    json_schema = Column(JSONB)
    instruction = Column(Text)
    user_id = Column(UUID, ForeignKey("users.id"))
    user = relationship("User", back_populates="extractors")
    extractor_examples = relationship("ExtractorExample", back_populates="extractor")

    def __repr__(self) -> str:
        return f"<Extractor(id={self.id}, description={self.description})>"



class User(SQLAlchemyBaseUserTableUUID, Base):  # type: ignore
    """
    Extended user model with additional fields like name, contact information, and address.
    Core of user-related operations, linked to applications, skills, experiences, etc.
    """
    __tablename__ = "users"
    first_name = Column(String)
    last_name = Column(String)
    phone_number = Column(String)
    address_line_1 = Column(String)
    address_line_2 = Column(String)
    city = Column(String)
    state = Column(String)
    zip_code = Column(String)
    country = Column(String)
    time_zone = Column(String)
    extractors = relationship("Extractor", back_populates="user")
    orchestration_pipelines = relationship(
        "OrchestrationPipeline", back_populates="user"
    )
