from pydantic import BaseModel, Field
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str = Field("ok", description="API status")
    model_loaded: bool
    mode: str  # 'pipeline' or 'separate'
    model_class: Optional[str] = None
    vectorizer_class: Optional[str] = None

class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of raw texts")

class PredictItem(BaseModel):
    text: str
    label: str
    proba: Optional[List[float]] = None

class PredictResponse(BaseModel):
    predictions: List[PredictItem]

class ExplainRequest(BaseModel):
    text: str

class TokenContribution(BaseModel):
    token: str
    weight: float

class ExplainResponse(BaseModel):
    tokens: List[TokenContribution]
    top_positive: List[TokenContribution]
    top_negative: List[TokenContribution]

class PredictOneRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=280, description="Texte court (<= 280 chars)")

class PredictOneResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float

class ExplainLimeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=280)

class ExplainLimeResponse(BaseModel):
    sentiment: str
    confidence: float
    explanation: List[str]        # liste de features/mots expliqués (labels lisibles)
    html_explanation: str         # HTML LIME
