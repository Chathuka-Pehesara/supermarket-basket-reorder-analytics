#!/usr/bin/env bash
# ===================================================
# Launch JupyterLab / Notebook directly on macOS / Linux
# ===================================================

set -e

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Create .venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Ensure dependencies are installed if needed
if ! python3 -c "import jupyterlab" &> /dev/null; then
    echo "Installing required dependencies..."
    pip install -r requirements.txt
fi

# 4. Launch JupyterLab (macOS automatically opens the default browser)
echo "Starting JupyterLab..."
jupyter lab
