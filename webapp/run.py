#!/usr/bin/env python
"""Web application startup script"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    
    # Set default model path if not set via environment variable
    if "MODEL_CKPT" not in os.environ:
        possible_paths = [
            os.path.join(project_root, "evo2_mix", "checkpoints", "best.ckpt"),
            os.path.join(project_root, "results", "best.ckpt"),
            os.path.join(project_root, "results", "checkpoints", "best.ckpt"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["MODEL_CKPT"] = os.path.abspath(path)
                break
    
    # Set default sequence length (3001bp = 2000bp upstream + TSS + 1000bp downstream)
    if "SEQ_LEN" not in os.environ:
        os.environ["SEQ_LEN"] = "3001"
    
    port = int(os.getenv("PORT", "8000"))
    
    print(f"\n🚀 Starting Web Server...")
    print(f"   URL: http://localhost:{port}")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        uvicorn.run(
            "webapp.api:app",
            host="0.0.0.0",
            port=port,
            reload=False,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install: pip install fastapi uvicorn pydantic numpy")
        sys.exit(1)
