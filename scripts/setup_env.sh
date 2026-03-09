#!/usr/bin/env bash
# scripts/setup_env.sh
# Automated development environment setup for PyTorch Mastery Hub
# Usage: bash scripts/setup_env.sh [--gpu] [--all]

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'  # No color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────
GPU_MODE=false
INSTALL_ALL=false

# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)     GPU_MODE=true; shift ;;
    --all)     INSTALL_ALL=true; shift ;;
    -h|--help)
      echo "Usage: bash scripts/setup_env.sh [--gpu] [--all]"
      echo "  --gpu   Install CUDA-enabled PyTorch (CUDA 11.8)"
      echo "  --all   Install all optional extras"
      exit 0 ;;
    *)
      warn "Unknown argument: $1"; shift ;;
  esac
done

# ── Header ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        PyTorch Mastery Hub — Environment Setup       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check Python ─────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 || "$PYTHON_MINOR" -lt 8 ]]; then
  error "Python 3.8+ required. Found: $PYTHON_VERSION"
fi
success "Python $PYTHON_VERSION"

# ── Create virtual environment ────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

if [[ -d "$VENV_DIR" ]]; then
  warn "Virtual environment already exists at $VENV_DIR"
  read -rp "Recreate it? [y/N] " response
  if [[ "$response" =~ ^[Yy] ]]; then
    rm -rf "$VENV_DIR"
    info "Removed existing venv."
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  success "Created venv at $VENV_DIR"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Activated virtual environment"

# ── Upgrade pip ───────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
success "pip upgraded"

# ── Install PyTorch ───────────────────────────────────────────────
if $GPU_MODE; then
  info "Installing PyTorch with CUDA 11.8 support..."
  pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 -q
  success "PyTorch (CUDA 11.8) installed"
else
  info "Installing PyTorch (CPU / MPS)..."
  pip install torch torchvision torchaudio -q
  success "PyTorch installed"
fi

# ── Install project ───────────────────────────────────────────────
cd "$PROJECT_ROOT"

if $INSTALL_ALL; then
  info "Installing all extras..."
  pip install -e ".[dev,notebooks,advanced,cv,nlp]" -q
else
  info "Installing dev + notebooks extras..."
  pip install -e ".[dev,notebooks]" -q
fi
success "Project installed"

# ── Pre-commit hooks ──────────────────────────────────────────────
info "Installing pre-commit hooks..."
pre-commit install
success "Pre-commit hooks installed"

# ── Verify ────────────────────────────────────────────────────────
echo ""
info "Verifying installation..."
python3 -c "
import torch, torchvision, numpy, matplotlib, pytest, jupyter
print(f'  PyTorch    : {torch.__version__}')
print(f'  TorchVision: {torchvision.__version__}')
print(f'  NumPy      : {numpy.__version__}')
DEVICE = 'mps' if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  Best device: {DEVICE}')
"
success "All packages imported successfully"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                  Setup Complete! 🎉                  ║"
echo "╟──────────────────────────────────────────────────────╢"
echo "║  Activate: source venv/bin/activate                  ║"
echo "║  Jupyter : jupyter lab notebooks/                    ║"
echo "║  Tests   : make test                                 ║"
echo "║  Help    : make help                                 ║"
echo "╚══════════════════════════════════════════════════════╝"
