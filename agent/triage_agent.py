"""
MediFlow Triage Agent (LangGraph)

image -> classify -> Grad-CAM -> confidence check -> retrieve guidelines
-> grade retrieval -> generate report | flag for review

Model: best_model_truecrop.pth -- frozen DenseNet121, true bounding-box
lung crops, no CLAHE. 84% NORMAL recall, 0.89 macro-F1.
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from gradcam.gradcam import (
    GradCAM,
    load_trained_model,
    preprocess_image,
    overlay_heatmap,
)
from rag.retriever import retrieve_guidelines

# ==========================
# Config
# ==========================

CHECKPOINT_PATH = "models/best_model_truecrop.pth"
DEVICE = "cpu"
CLASSES = ["NORMAL", "PNEUMONIA"]
CONFIDENCE_THRESHOLD = 0.75
GRADCAM_OUTPUT_DIR = "agent_outputs"
RETRIEVAL_K = 4

RETRIEVAL_QUERIES = {
    "NORMAL": "normal chest x-ray assessment pediatric pneumonia exclusion criteria",
    "PNEUMONIA": "pediatric community-acquired pneumonia diagnosis chest x-ray findings management",
}

_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class RetrievalGrade(BaseModel):
    sufficient: bool = Field(
        description="True if retrieved chunks contain genuinely relevant, "
                    "substantive clinical guidance -- not just topically-adjacent "
                    "text, reference lists, or citations."
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


class TriageReport(BaseModel):
    finding: str = Field(description="1-2 sentence summary of the imaging finding")
    clinical_context: str = Field(description="Relevant guidance from retrieved chunks, cited by source")
    recommended_next_steps: str = Field(description="Suggested next steps, general not prescriptive")
    disclaimer: str = Field(description="Standard AI-assisted tool disclaimer")


_grading_llm = _llm.with_structured_output(RetrievalGrade)
_report_llm = _llm.with_structured_output(TriageReport)


# ==========================
# State
# ==========================

class TriageState(TypedDict):
    image_path: str
    prediction: str
    confidence: float
    gradcam_path: str
    needs_review: bool
    retrieved_chunks: list
    retrieval_sufficient: bool
    retrieval_reasoning: str
    status: str


# ==========================
# Model loading (once, at import time)
# ==========================

_model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)
_target_layer = _model.features.norm5
_gradcam = GradCAM(_model, _target_layer)


# ==========================
# Nodes
# ==========================

def load_and_classify(state: TriageState) -> dict:
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
    return {"needs_review": state["confidence"] < CONFIDENCE_THRESHOLD}


def flag_for_review(state: TriageState) -> dict:
    status = (
        f"FLAGGED FOR REVIEW: {state['prediction']} predicted at "
        f"{state['confidence']:.1%} confidence (below "
        f"{CONFIDENCE_THRESHOLD:.0%} threshold). Recommend radiologist review."
    )
    return {"status": status}


def retrieve_guidelines_node(state: TriageState) -> dict:
    query = RETRIEVAL_QUERIES[state["prediction"]]
    chunks = retrieve_guidelines(query, k=RETRIEVAL_K)
    return {"retrieved_chunks": chunks}


def grade_retrieval(state: TriageState) -> dict:
    chunks_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in state["retrieved_chunks"]
    )

    prompt = (
        f"Chest X-ray triage result: {state['prediction']} "
        f"(confidence: {state['confidence']:.1%}).\n\n"
        f"Write 'finding' as a radiology-style findings statement (like: "
        f"'No evidence of consolidation, effusion, or pneumothorax. Cardiac "
        f"silhouette and mediastinum within normal limits.') -- do NOT "
        f"discuss the AI tool, guideline citations, or imaging modality "
        f"comparisons in this field.\n\n"
        f"Use the retrieved context below ONLY for 'clinical_context' "
        f"(brief, 1-2 sentences, cite source) and 'recommended_next_steps'.\n\n"
        f"Retrieved context:\n{chunks_text}")

    grade: RetrievalGrade = _grading_llm.invoke(prompt)
    return {
        "retrieval_sufficient": grade.sufficient,
        "retrieval_reasoning": grade.reasoning,
    }


def flag_insufficient_context(state: TriageState) -> dict:
    status = (
        f"FLAGGED - INSUFFICIENT CONTEXT: {state['prediction']} predicted "
        f"at {state['confidence']:.1%} confidence, but retrieved guideline "
        f"content was judged insufficient to ground a report. "
        f"Reasoning: {state['retrieval_reasoning']} "
        f"Recommend radiologist review."
    )
    return {"status": status}


def generate_report(state: TriageState) -> dict:
    chunks_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in state["retrieved_chunks"]
    )

    prompt = (
    f"Chest X-ray triage result: {state['prediction']} "
    f"(confidence: {state['confidence']:.1%}).\n\n"
    f"Write 'finding' as bullet-point radiology findings, one line per "
    f"anatomical region, terse clinical style:\n"
    f"- Pleura (effusion/pneumothorax)\n"
    f"- Cardiac silhouette / mediastinum\n"
    f"- Lung fields (consolidation/opacity -- location if PNEUMONIA)\n"
    f"- Bones/soft tissue\n\n"
    f"Example NORMAL style: 'No evidence of pleural effusion or "
    f"pneumothorax. Cardiac silhouette and mediastinum within normal "
    f"limits. Lung fields clear. Bony structures and soft tissues "
    f"unremarkable.'\n\n"
    f"Example PNEUMONIA style: 'No pleural effusion or pneumothorax. "
    f"Cardiac silhouette normal. Focal consolidation noted, consistent "
    f"with pneumonia. Bony structures intact.'\n\n"
    f"Do NOT mention the AI tool, confidence, citations, or imaging-"
    f"modality comparisons in 'finding'.\n\n"
    f"'clinical_context' and 'recommended_next_steps' must relate "
    f"directly to THIS case's prediction ({state['prediction']}) -- "
    f"omit retrieved chunks that aren't directly applicable rather "
    f"than including them anyway.\n\n"
    f"'disclaimer' must state this is an AI-assisted screening tool, not "
    f"a diagnosis, and that clinician review is required regardless of "
    f"confidence level.\n\n"
    f"Retrieved context:\n{chunks_text}"
)

    report: TriageReport = _report_llm.invoke(prompt)

    status = (
        f"REPORT GENERATED\n"
        f"Finding: {report.finding}\n"
        f"Clinical Context: {report.clinical_context}\n"
        f"Next Steps: {report.recommended_next_steps}\n"
        f"Disclaimer: {report.disclaimer}"
    )
    return {"status": status}


# ==========================
# Conditional routing
# ==========================

def route_after_confidence_check(state: TriageState) -> str:
    if state["needs_review"]:
        return "flag_for_review"
    return "retrieve_guidelines_node"


def route_after_retrieval_grade(state: TriageState) -> str:
    if state["retrieval_sufficient"]:
        return "generate_report"
    return "flag_insufficient_context"


# ==========================
# Build the graph
# ==========================

def build_triage_graph():
    graph = StateGraph(TriageState)

    graph.add_node("load_and_classify", load_and_classify)
    graph.add_node("check_confidence", check_confidence)
    graph.add_node("flag_for_review", flag_for_review)
    graph.add_node("retrieve_guidelines_node", retrieve_guidelines_node)
    graph.add_node("grade_retrieval", grade_retrieval)
    graph.add_node("flag_insufficient_context", flag_insufficient_context)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("load_and_classify")
    graph.add_edge("load_and_classify", "check_confidence")

    graph.add_conditional_edges(
        "check_confidence",
        route_after_confidence_check,
        {
            "flag_for_review": "flag_for_review",
            "retrieve_guidelines_node": "retrieve_guidelines_node",
        },
    )

    graph.add_edge("retrieve_guidelines_node", "grade_retrieval")

    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_retrieval_grade,
        {
            "generate_report": "generate_report",
            "flag_insufficient_context": "flag_insufficient_context",
        },
    )

    graph.add_edge("flag_for_review", END)
    graph.add_edge("flag_insufficient_context", END)
    graph.add_edge("generate_report", END)

    return graph.compile()


if __name__ == "__main__":
    IMAGE_PATH = "sample_images_by_class/PNEUMONIA/person372_bacteria_1706.jpeg"

    triage_app = build_triage_graph()
    result = triage_app.invoke({"image_path": IMAGE_PATH})

    print(f"Prediction     : {result['prediction']}")
    print(f"Confidence     : {result['confidence']:.2%}")
    print(f"Needs review   : {result['needs_review']}")
    print(f"Grad-CAM saved : {result['gradcam_path']}")
    print(f"Status         : {result['status']}")