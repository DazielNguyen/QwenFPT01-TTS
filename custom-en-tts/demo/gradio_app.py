"""Gradio demo: 6-voice English TTS.

Voices: Ryan, Aiden, Emma, Sophia, Oliver, Isabella

Security requirements (phase-07 plan):
    - Terms-of-use accept gate on first load
    - Server-side rate limiting: max 10 requests/minute per IP
    - Model card link with allowed/disallowed uses
    - No voice cloning for impersonation of real persons

Latency target:
    - 20-word sentence < 3s on A10G
    - < 5s on T4 fallback

Usage:
    python demo/gradio_app.py --checkpoint checkpoints/speaker_sft/

Secrets required (env vars):
    None for inference. HF_TOKEN only needed if loading from Hub.
"""

raise NotImplementedError("Implement in Phase 7 Sprint 12")
