"""Reference integration snippet for fastapiCollector/main.py.

This file mirrors the exact logic integrated into backend main.py and is kept here
as a standalone reference requested for the LSTM folder deliverables.
"""

INTEGRATION_NOTES = {
    "startup": "Load BehavioralInference once and keep statistical fallback.",
    "enroll": "Compute DNA embedding and store in templates.stats_vector as JSON object with both stats and fused_dna.",
    "verify": "Prefer ONNX DNA risk; fallback to old cosine stats if artifacts/model unavailable.",
}
