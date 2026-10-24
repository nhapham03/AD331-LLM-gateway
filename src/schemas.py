from pydantic import BaseModel, ConfigDict, Field


class ChunkTelemetry(BaseModel):
    """Request-level token and latency metrics, emitted on the final chunk."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_latency_seconds: float = Field(ge=0)
    ttft_seconds: float = Field(ge=0)
    tokens_per_second: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    estimated_input_tokens: int | None = None
    usage_is_estimated: bool


class StreamChunk(BaseModel):
    """A provider-independent text chunk."""

    model_config = ConfigDict(frozen=True)

    delta_text: str
    provider: str
    model: str
    is_final: bool
    telemetry: ChunkTelemetry | None = None