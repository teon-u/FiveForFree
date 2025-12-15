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
import socket
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
            return False
        except OSError:
            return True


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use(host, port):
            return port
    return -1


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

    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Automatically find available port if default is in use",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.")
        print("Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

    # Check port availability
    port = args.port
    if is_port_in_use(args.host, port):
        if args.auto_port:
            new_port = find_available_port(args.host, port)
            if new_port == -1:
                print(f"ERROR: No available ports found starting from {port}")
                sys.exit(1)
            print(f"WARNING: Port {port} is in use, using port {new_port} instead")
            port = new_port
        else:
            print("=" * 60)
            print(f"ERROR: Port {port} is already in use!")
            print("=" * 60)
            print()
            print("Possible solutions:")
            print(f"  1. Use a different port: python run_api.py --port {port + 1}")
            print(f"  2. Auto-select port: python run_api.py --auto-port")
            print(f"  3. Find and stop the process using port {port}:")
            print(f"     - Windows: netstat -ano | findstr :{port}")
            print(f"     - Linux/Mac: lsof -i :{port}")
            print()
            sys.exit(1)

    print("=" * 60)
    print("NASDAQ Prediction API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    print()
    print("Starting server...")
    print(f"API docs: http://{args.host}:{port}/docs")
    print(f"WebSocket: ws://{args.host}:{port}/ws")
    print()

    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1,  # Can't use workers with reload
    )


if __name__ == "__main__":
    main()
