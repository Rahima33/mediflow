from pydantic import BaseModel


class TriageResponse(BaseModel):
    prediction: str
    confidence: float
    needs_review: bool
    gradcam_base64: str
    finding: str
    clinical_context: str
    next_steps: str
    disclaimer: str