"""
Hugging Face Spaces entry point for Skin Disease Detection API.
Exposes the FastAPI app via uvicorn on port 7860 (required by HF Spaces).
"""
import os
import subprocess
import sys

# HF Spaces requires port 7860
PORT = int(os.getenv("PORT", 7860))

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api:app",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--workers", "1",
        "--log-level", "info",
    ])
