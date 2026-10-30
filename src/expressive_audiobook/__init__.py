"""Book (PDF, Markdown or text) to expressive audiobook pipeline.

Stages read and write artifacts on disk so each stage can be rerun alone.
Text prep and speech synthesis call Gemini when ``GEMINI_API_KEY`` is set.
"""

__version__ = "0.1.0"
