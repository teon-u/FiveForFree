#!/usr/bin/env python3
"""
Startup script for the NASDAQ Prediction API.

This script starts the FastAPI server with production-ready settings.

Usage:
    python run_api.py

    Or with custom settings:
    python run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Run the FastAPI application."""
    parser = argparse.ArgumentParser(description="Start NASDAQ Prediction API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.")
        print("Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

    print("=" * 60)
    print("NASDAQ Prediction API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    print()
    print("Starting server...")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"WebSocket: ws://{args.host}:{args.port}/ws")
    print()

    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1,  # Can't use workers with reload
    )


if __name__ == "__main__":
    main()
