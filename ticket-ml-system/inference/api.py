import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from inference.predictor import TicketPredictor, load_predictor

logger = logging.getLogger(__name__)


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, description="Support ticket text")
    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class BatchTicketRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]
    is_confident: bool
    latency_ms: Optional[float] = None

predictor: Optional[TicketPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    predictor = load_predictor()
    logger.info("Model ready.")
    yield
    predictor = None


app = FastAPI(
    title="Ticket Classifier API",
    description="Classifies support tickets into billing, technical_issue, account_access, refund, feature_request",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Latency-Ms"] = f"{elapsed:.1f}"
    return response

@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["classify"])
async def predict_single(request: TicketRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    result = predictor.predict(request.text)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        all_scores=result.all_scores,
        is_confident=result.is_confident,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", tags=["classify"])
async def predict_batch(request: BatchTicketRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    results = predictor.predict_batch(request.texts)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "predictions": [
            {
                "label": r.label,
                "confidence": r.confidence,
                "is_confident": r.is_confident,
            }
            for r in results
        ],
        "total_latency_ms": latency_ms,
        "count": len(results),
    }