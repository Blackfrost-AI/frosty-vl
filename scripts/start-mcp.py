#!/usr/bin/env python3
"""Absolute-path MCP launcher for clients that do not set a working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frosty_mcp.__main__ import main

if __name__ == '__main__':
    main()
