#!/bin/bash
# Wrapper script to run llm_controller.py with the venv Python that has google-genai installed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/krishnan_kkrish_altostrat_com/llm_venv/bin/python3"

exec "$VENV_PYTHON" "$SCRIPT_DIR/llm_controller.py" "$@"
