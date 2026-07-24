"""
MediFlow Triage Agent (LangGraph)

Orchestrates the pipeline stages after the raw model exists:
    image -> classify -> Grad-CAM -> confidence check -> (flag for review | generate report)

This is v1 of the graph: report generation is a stub, to be replaced once
the RAG + LLM report-generation stage is built. The goal right now is a
correct, working graph structure with real classification + Grad-CAM
nodes and real conditional routing.

Model: best_model_truecrop.pth -- frozen DenseNet121 backbone, trained on
true bounding-box lung crops (no CLAHE). This is the best-performing and
most thoroughly validated checkpoint from the MediFlow investigation:
84% NORMAL recall, 0.89 macro-F1, and Grad-CAM-confirmed reduction of the
mask-boundary shortcut present in earlier lung-crop attempts.
"""

import os
from typing import TypedDict

import torch
from PIL import Image
from langgraph.graph import StateGraph, END

from gradcam.gradcam import (
    GradCAM,
    load_trained_model,
    preprocess_image,
    overlay_heatmap,
)

# ==========================
# Config
# ==========================

CHECKPOINT_PATH = "models/best_model_truecrop.pth"
DEVICE = "cpu"
CLASSES = ["NORMAL", "PNEUMONIA"]

# Below this confidence, regardless of predicted class, the case gets
# flagged for human review instead of proceeding straight to a report.
# A single global threshold is a simple v1 choice -- a more clinically
# careful version might use a stricter threshold specifically for
# NORMAL predictions, since a missed PNEUMONIA case (false reassurance)
# is more costly than an unnecessary review flag.
CONFIDENCE_THRESHOLD = 0.75

GRADCAM_OUTPUT_DIR = "agent_outputs"


# ==========================
# State
# ==========================

class TriageState(TypedDict):
    image_path: str
    prediction: str
    confidence: float
    gradcam_path: str
    needs_review: bool
    status: str


# ==========================
# Model loading (once, at import time -- not per-request)
# ==========================

_model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)
_target_layer = _model.features.norm5
_gradcam = GradCAM(_model, _target_layer)


# ==========================
# Nodes
# ==========================

def load_and_classify(state: TriageState) -> dict:
    """
    Load the X-ray, run it through the classifier, and record the
    prediction + confidence. This also runs the forward+backward pass
    needed for Grad-CAM, so we generate the heatmap in the same node
    rather than re-running the model twice.
    """
    original_image, input_tensor = preprocess_image(state["image_path"])

    cam, predicted_class, confidence = _gradcam.generate(input_tensor)

    os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)
    overlay = overlay_heatmap(original_image, cam)

    base_name = os.path.splitext(os.path.basename(state["image_path"]))[0]
    gradcam_path = os.path.join(GRADCAM_OUTPUT_DIR, f"{base_name}_gradcam.png")
    Image.fromarray(overlay).save(gradcam_path)

    return {
        "prediction": CLASSES[predicted_class],
        "confidence": confidence,
        "gradcam_path": gradcam_path,
    }


def check_confidence(state: TriageState) -> dict:
    """
    Decide whether this case needs human review. This node itself just
    records the decision in state; the actual branching happens in the
    conditional edge below (route_after_confidence_check), which reads
    this same field.
    """
    needs_review = state["confidence"] < CONFIDENCE_THRESHOLD
    return {"needs_review": needs_review}


def flag_for_review(state: TriageState) -> dict:
    """Placeholder terminal node for low-confidence cases."""
    status = (
        f"FLAGGED FOR REVIEW: {state['prediction']} predicted at "
        f"{state['confidence']:.1%} confidence (below "
        f"{CONFIDENCE_THRESHOLD:.0%} threshold). Recommend radiologist review."
    )
    return {"status": status}


def generate_report(state: TriageState) -> dict:
    """
    Placeholder for the future RAG + LLM report-generation stage.
    For now, just confirms the case would proceed to report generation.
    """
    status = (
        f"PROCEEDING TO REPORT: {state['prediction']} predicted at "
        f"{state['confidence']:.1%} confidence. "
        f"(Report generation not yet implemented -- next pipeline stage.)"
    )
    return {"status": status}


# ==========================
# Conditional routing
# ==========================

def route_after_confidence_check(state: TriageState) -> str:
    """
    Tells LangGraph which node to run next, based on state.
    Return value must match a key in the conditional edge mapping below.
    """
    if state["needs_review"]:
        return "flag_for_review"
    return "generate_report"


# ==========================
# Build the graph
# ==========================

def build_triage_graph():
    graph = StateGraph(TriageState)

    graph.add_node("load_and_classify", load_and_classify)
    graph.add_node("check_confidence", check_confidence)
    graph.add_node("flag_for_review", flag_for_review)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("load_and_classify")
    graph.add_edge("load_and_classify", "check_confidence")

    graph.add_conditional_edges(
        "check_confidence",
        route_after_confidence_check,
        {
            "flag_for_review": "flag_for_review",
            "generate_report": "generate_report",
        },
    )

    graph.add_edge("flag_for_review", END)
    graph.add_edge("generate_report", END)

    return graph.compile()


if __name__ == "__main__":
    IMAGE_PATH = "sample_images_by_class/PNEUMONIA/person23_bacteria_93.jpeg"  # swap in a real test image

    triage_app = build_triage_graph()

    result = triage_app.invoke({"image_path": IMAGE_PATH})

    print(f"Prediction     : {result['prediction']}")
    print(f"Confidence     : {result['confidence']:.2%}")
    print(f"Needs review   : {result['needs_review']}")
    print(f"Grad-CAM saved : {result['gradcam_path']}")
    print(f"Status         : {result['status']}")