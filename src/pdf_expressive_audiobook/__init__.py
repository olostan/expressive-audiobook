"""PDF to expressive audiobook pipeline.

Stages read and write artifacts on disk so each stage can be rerun alone.
Stages 1 and 2 are implemented. Stage 3 validates and builds multimodal
requests. Stage 4 plans TTS requests. Stage 5 concatenates audio. The network
calls for text prep and synthesis are stubbed.
"""

__version__ = "0.1.0"
