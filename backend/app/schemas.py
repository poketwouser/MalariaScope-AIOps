"""Response schemas: the contract with the frontend (LLD section 3)."""

from typing import List, Literal

from pydantic import BaseModel, Field

Label = Literal["parasitized", "uninfected"]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str


class PredictResponse(BaseModel):
    request_id: str
    label: Label
    is_parasitized: bool
    p_parasitized: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1, description="probability of `label`")
    threshold: float
    model_version: str
    latency_ms: float


class FeedbackResponse(BaseModel):
    stored: bool
    feedback_id: int
    agreed: bool
    feedback_count: int


class FeedbackStats(BaseModel):
    total: int
    agreed: int
    disagreed: int
    since_last_export: int


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str
    threshold: float
    image_size: int
    test_accuracy: float
    test_recall_parasitized: float
    git_commit: str
    data_md5: str
    class_names: List[str]


class Health(BaseModel):
    status: str


class Ready(BaseModel):
    status: str
    model_server: str
