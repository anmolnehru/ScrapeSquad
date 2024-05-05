# Path: app/schemas.py
import json
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path  # TODO: Use Literal for performance improvement
from typing import Any, Literal, Sequence, TypeVar

from fastapi import UploadFile
from fastapi_users import schemas
from pydantic import UUID4, AnyHttpUrl
from pydantic import BaseModel as _BaseModel
from pydantic import EmailStr, Field, model_validator, validator
from PyPDF2 import PdfReader

from app import utils


# Base Model
class BaseSchema(_BaseModel):
    class Config:
        from_attributes = True
        protected_namespaces = ()  # Setting protected namespaces to empty


# Types, properties, and shared models


BaseSchemaSubclass = TypeVar("BaseSchemaSubclass", bound=BaseSchema)


class OrchestrationEventStatusType(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failure"


class URIType(str, Enum):
    FILE = "filepath"
    DATALAKE = "datalake"
    DATABASE = "database"
    API = "api"
    URL = "url"


class URI(BaseSchema):
    name: str
    type: URIType

    class Config:
        json_encoders = {
            "URI": lambda v: v.dict(),
        }

    @validator("type", pre=True)  # Not sure if pre=True is neccessary?
    def validate_type(cls, v: str) -> URIType:
        return URIType(v)


class BaseRead(BaseSchema):
    id: UUID4 = Field(description="The unique uuid4 record identifier.")
    created_at: datetime = Field(description="The time the item was created")
    updated_at: datetime = Field(description="The time the item was last updated")


class Pagination(BaseSchema):
    page: int = Field(1, ge=1, description="The page number")
    page_size: int = Field(10, ge=1, description="The number of items per page")
    request_count: bool = Field(False, description="Request a query for total count")


# Model CRUD Schemas
class BaseOrchestrationPipeline(BaseSchema):
    name: str | None = Field(None, description="Name of the pipeline")
    description: str | None = Field(None, description="Description of the pipeline")
    definition: dict | None = Field(None, description="Parameters for the pipeline")


class OrchestrationPipelineRead(BaseOrchestrationPipeline, BaseRead):
    events: list["OrchestrationEventRead"] = Field(
        [], description="Events in the pipeline", alias="orchestration_events"
    )


class OrchestrationPipelineCreate(BaseOrchestrationPipeline):
    pass


class OrchestrationPipelineUpdate(BaseOrchestrationPipeline):
    events: list["OrchestrationEventRead"] = Field(
        [], description="Events in the pipeline"
    )


class BaseOrchestrationEvent(BaseSchema):
    message: str | None = Field(None, description="Error message")
    payload: dict | None = Field(None, description="Payload of the triggering event")
    environment: str | None = Field(None, description="Application environment setting")
    source_uri: URI | None = Field(None, description="Source of the pipeline")
    destination_uri: URI | None = Field(None, description="Destination of the pipeline")
    status: OrchestrationEventStatusType | None = Field(
        None, description="Status of the event"
    )
    pipeline_id: UUID4 | None = Field(None, description="Pipeline ID")

    @validator("source_uri", "destination_uri", pre=True)
    def validate_uri(cls, v: Any) -> URI | None:
        if isinstance(v, str):
            return URI(**json.loads(v))
        if isinstance(v, dict):
            return URI(**v)
        return v


class OrchestrationEventRead(BaseOrchestrationEvent, BaseRead):
    @validator("payload", pre=True)
    def load_json(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Payload must be a valid JSON")
        return v


class OrchestrationEventCreate(BaseOrchestrationEvent):
    pipeline_id: UUID4


class OrchestrationEventUpdate(BaseOrchestrationEvent):
    pass


class ExtractorRequest(BaseSchema):
    llm_name: str | None = Field("gpt-3.5-turbo", description="Model name")
    examples: list["ExtractorExampleRead"] = Field(
        [], description="Extraction examples"
    )
    instructions: str | None = Field(None, description="Extraction instruction")
    json_schema: dict | None = Field(None, description="JSON schema", alias="schema")
    text: str | None = Field(None, description="Text to extract from")

    @validator("json_schema")
    def validate_schema(cls, v: Any) -> dict[str, Any]:
        """Validate the schema."""
        utils.validate_json_schema(v)
        return v


class ExtractorResponse(BaseSchema):
    data: list[Any] = Field([], description="Extracted data")
    content_too_long: bool = Field(False, description="Content too long to extract")


class BaseExtractorExample(BaseSchema):
    content: str | None = Field(None, description="Example content")
    output: str | None = Field(None, description="Example output")


class ExtractorExampleRead(BaseRead, BaseExtractorExample):
    pass


class ExtractorExampleCreate(BaseExtractorExample):
    pass


class ExtractorExampleUpdate(BaseExtractorExample):
    pass


class BaseExtractor(BaseSchema):
    name: str | None = Field(None, description="Extractor name")
    description: str | None = Field(None, description="Extractor description")
    json_schema: dict | str | None = Field(None, description="JSON schema")
    instruction: str | None = Field(None, description="Extractor instruction")
    extractor_examples: list[ExtractorExampleRead] = Field(
        [], description="Extractor examples"
    )

    @validator("json_schema")
    def validate_schema(cls, v: Any) -> dict[str, Any]:
        """Validate the schema."""
        if isinstance(v, str):
            v = json.loads(v)
        if v:
            utils.validate_json_schema(v)
        return v


class ExtractorRead(BaseRead, BaseExtractor):
    pass


class ExtractorCreate(BaseExtractor):
    pass


class ExtractorUpdate(BaseExtractor):
    pass


class ExtractorRun(BaseSchema):
    """Request to run an extractor."""

    mode: Literal["entire_document", "retrieval"] = Field(
        "entire_document",
        description="Mode to run the extractor in. 'entire_document' extracts information from the entire document. 'retrieval' extracts information from a specific section of the document.",
    )
    file: UploadFile | None = Field(
        None,
        description="A file to extract information from. If provided, the file will be processed and the text extracted.",
    )
    text: str | None = Field(
        None,
        description="Text to extract information from. If provided, the text will be processed and the information extracted.",
    )
    url: AnyHttpUrl | None = Field(
        None,
        description="A URL to extract information from. If provided, the URL will be processed and the information extracted.",
    )
    llm: str | None = Field(
        None,
        description="The language model to use for the extraction.",
    )



class BaseUser(BaseSchema):
    first_name: str | None = Field(None, description="First name")
    last_name: str | None = Field(None, description="Last name")
    phone_number: str | None = Field(None, description="Phone number")
    address_line_1: str | None = Field(None, description="Address line 1")
    address_line_2: str | None = Field(None, description="Address line 2")
    city: str | None = Field(None, description="City")
    state: str | None = Field(None, description="State")
    zip_code: str | None = Field(None, description="Zip code")
    country: str | None = Field(None, description="Country")
    time_zone: str | None = Field(None, description="Time zone")
    avatar_uri: URI | None = Field(None, description="Avatar URI")


class UserRead(schemas.BaseUser[UUID4], BaseUser):  # type: ignore
    pass

class UserCreate(schemas.BaseUserCreate, BaseUser):
    pass


class UserUpdate(schemas.BaseUserUpdate, BaseUser):
    pass
