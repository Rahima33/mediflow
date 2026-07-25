import base64
import tempfile
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from agent.triage_agent import build_triage_graph
from api.schemas import TriageResponse

app = FastAPI(title="MediFlow Triage API")

frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
allow_origins = [origin.strip() for origin in frontend_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup, not per-request
_triage_app = build_triage_graph()


@app.post("/triage", response_model=TriageResponse)
async def triage(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _triage_app.invoke({"image_path": tmp_path})

        with open(result["gradcam_path"], "rb") as f:
            gradcam_b64 = base64.b64encode(f.read()).decode("utf-8")

        return TriageResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            needs_review=result["needs_review"],
            gradcam_base64=gradcam_b64,
            finding=result["finding"],
            clinical_context=result["clinical_context"],
            next_steps=result["next_steps"],
            disclaimer=result["disclaimer"],
        )
    finally:
        os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}