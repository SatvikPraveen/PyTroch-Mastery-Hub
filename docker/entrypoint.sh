#!/bin/bash
# docker/entrypoint.sh
# Container entrypoint script with comprehensive setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Environment variables with defaults
ENVIRONMENT=${ENVIRONMENT:-production}
LOG_LEVEL=${LOG_LEVEL:-INFO}
PYTHONPATH=${PYTHONPATH:-/app}
WORKERS=${WORKERS:-4}
TIMEOUT=${TIMEOUT:-300}
MAX_REQUESTS=${MAX_REQUESTS:-1000}
MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}

log "Starting ML Training Framework container"
log "Environment: ${ENVIRONMENT}"
log "Python path: ${PYTHONPATH}"
log "Log level: ${LOG_LEVEL}"

# Create necessary directories
log "Creating required directories..."
mkdir -p /app/{logs,data,models,checkpoints,results}
mkdir -p /app/data/{raw,processed,external,cache}
mkdir -p /app/results/{figures,reports,experiments}

# Set proper permissions
if [ "$(id -u)" = "0" ]; then
    warn "Running as root, setting up permissions..."
    chown -R mluser:mluser /app/logs /app/data /app/models /app/checkpoints /app/results 2>/dev/null || true
fi

# Health check function
health_check() {
    log "Performing health checks..."
    
    # Python import check
    if ! python -c "import torch, numpy, pandas, sklearn" 2>/dev/null; then
        error "Failed to import required Python packages"
        return 1
    fi
    
    # CUDA availability check (if GPU)
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU detected, checking CUDA availability..."
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
    fi
    
    # Database connection check (if configured)
    if [ -n "${DATABASE_URL}" ]; then
        log "Checking database connection..."
        timeout=30
        while [ $timeout -gt 0 ]; do
            if python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect('${DATABASE_URL}')
    conn.close()
    print('Database connection successful')
    exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" 2>/dev/null; then
                break
            fi
            sleep 2
            timeout=$((timeout-2))
        done
        
        if [ $timeout -le 0 ]; then
            warn "Database connection timeout, proceeding without database"
        fi
    fi
    
    # Redis connection check (if configured)
    if [ -n "${REDIS_URL}" ]; then
        log "Checking Redis connection..."
        if ! python -c "
import redis
import os
try:
    r = redis.from_url('${REDIS_URL}')
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
" 2>/dev/null; then
            warn "Redis connection failed, proceeding without Redis"
        fi
    fi
    
    success "Health checks completed"
}

# Environment-specific setup
setup_environment() {
    case "${ENVIRONMENT}" in
        "development"|"dev")
            log "Setting up development environment..."
            export PYTHONUNBUFFERED=1
            export PYTHONDONTWRITEBYTECODE=1
            export CUDA_LAUNCH_BLOCKING=1
            export TORCH_USE_CUDA_DSA=1
            ;;
        "testing"|"test")
            log "Setting up testing environment..."
            export PYTHONUNBUFFERED=1
            export PYTHONDONTWRITEBYTECODE=1
            export TESTING=1
            ;;
        "production"|"prod")
            log "Setting up production environment..."
            export PYTHONUNBUFFERED=1
            export PYTHONOPTIMIZE=1
            export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
            export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
            ;;
        *)
            warn "Unknown environment: ${ENVIRONMENT}, using production defaults"
            export PYTHONUNBUFFERED=1
            ;;
    esac
}

# Database migrations (if needed)
run_migrations() {
    if [ -n "${DATABASE_URL}" ] && [ -f "/app/migrations/migrate.py" ]; then
        log "Running database migrations..."
        python /app/migrations/migrate.py || warn "Migration failed, continuing..."
    fi
}

# Model initialization
initialize_models() {
    if [ -f "/app/scripts/init_models.py" ]; then
        log "Initializing models..."
        python /app/scripts/init_models.py || warn "Model initialization failed"
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    log "Shutting down gracefully..."
    # Kill any background processes
    jobs -p | xargs -r kill
    # Additional cleanup if needed
    if [ -f "/app/scripts/cleanup.py" ]; then
        python /app/scripts/cleanup.py
    fi
    success "Cleanup completed"
    exit 0
}

# Set trap for graceful shutdown
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Executing entrypoint script..."
    
    # Run setup
    setup_environment
    health_check
    run_migrations
    initialize_models
    
    # Handle different command types
    if [ $# -eq 0 ]; then
        warn "No command provided, starting default application"
        exec python -m training.train
    elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
        log "Starting shell..."
        exec "$@"
    elif [ "$1" = "python" ]; then
        log "Starting Python application: ${*:2}"
        exec "$@"
    elif [ "$1" = "jupyter" ]; then
        log "Starting Jupyter server..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app
    elif [ "$1" = "mlflow" ]; then
        log "Starting MLflow server..."
        exec mlflow server --host 0.0.0.0 --port 5000 "${@:2}"
    elif [ "$1" = "tensorboard" ]; then
        log "Starting TensorBoard..."
        exec tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006 "${@:2}"
    elif [ "$1" = "train" ]; then
        log "Starting training process..."
        exec python -m training.train "${@:2}"
    elif [ "$1" = "evaluate" ]; then
        log "Starting evaluation process..."
        exec python -m evaluation.evaluate "${@:2}"
    elif [ "$1" = "api" ]; then
        log "Starting API server..."
        if command -v gunicorn >/dev/null 2>&1; then
            exec gunicorn \
                --workers "${WORKERS}" \
                --timeout "${TIMEOUT}" \
                --max-requests "${MAX_REQUESTS}" \
                --max-requests-jitter "${MAX_REQUESTS_JITTER}" \
                --bind 0.0.0.0:8000 \
                --access-logfile - \
                --error-logfile - \
                --log-level "${LOG_LEVEL,,}" \
                "api.app:create_app()" "${@:2}"
        else
            exec python -m api.app "${@:2}"
        fi
    elif [ "$1" = "worker" ]; then
        log "Starting background worker..."
        exec python -m workers.worker "${@:2}"
    elif [ "$1" = "scheduler" ]; then
        log "Starting task scheduler..."
        exec python -m workers.scheduler "${@:2}"
    elif [ "$1" = "test" ]; then
        log "Running tests..."
        exec python -m pytest "${@:2}"
    elif [ "$1" = "lint" ]; then
        log "Running linting..."
        exec flake8 . "${@:2}"
    elif [ "$1" = "format" ]; then
        log "Running code formatting..."
        exec black . "${@:2}"
    else
        log "Executing custom command: $*"
        exec "$@"
    fi
}

# Execute main function
main "$@"