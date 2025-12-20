# Docker Deployment Guide

## Overview

Docker containerization for the unified AI/ML pipeline system.

## Docker Configuration

### docker-compose.unified.yml

**Services**:
- `unified-pipeline`: Main pipeline container

**Configuration**:
- Memory limits: 8GB (max), 4GB (reserved)
- CPU limits: 4 cores (max), 2 cores (reserved)
- Volumes: Repository (ro), knowledge_base, unified_output, datasets, trained_models

### Dockerfile.unified-pipeline

**Base Image**: `python:3.11-slim`

**Dependencies**:
- System: git, build-essential, cmake
- Python: numpy, torch, torchaudio, librosa, soundfile, pandas, pretty_midi, scikit-learn, matplotlib

**Features**:
- Knowledge extraction
- Training pipeline preparation
- Framework integration

## Quick Start

### Build Container

```bash
cd "/Users/seanburdges/Desktop/final kel"
docker-compose -f docker-compose.unified.yml build
```

### Run Pipeline

```bash
# Run in foreground
docker-compose -f docker-compose.unified.yml up unified-pipeline

# Run in background
docker-compose -f docker-compose.unified.yml up -d unified-pipeline

# Check logs
docker-compose -f docker-compose.unified.yml logs -f unified-pipeline
```

### Stop Container

```bash
docker-compose -f docker-compose.unified.yml down
```

## Configuration

### Environment Variables

Edit `docker-compose.unified.yml` to configure:

```yaml
environment:
  - ENABLE_KNOWLEDGE_EXTRACTION=true   # Enable/disable knowledge scan
  - ENABLE_TRAINING=true               # Enable/disable training prep
  - ENABLE_CIF_INTEGRATION=true        # Enable/disable CIF integration
  - ENABLE_LAS_INTEGRATION=true        # Enable/disable LAS integration
  - ENABLE_QEF=true                    # Enable/disable QEF
  - ENABLE_ETHICS=true                 # Enable/disable ethics framework
  - TRAINING_EPOCHS=50                 # Training epochs
  - TRAINING_BATCH_SIZE=32             # Batch size
  - TRAINING_DEVICE=cpu                # cpu, cuda, or mps
```

### Resource Limits

Adjust in `docker-compose.unified.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Increase for faster processing
      memory: 16G    # Increase for larger datasets
```

## GPU Support (Optional)

For NVIDIA GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then run:
```bash
docker-compose -f docker-compose.unified.yml run --rm \
  -e TRAINING_DEVICE=cuda \
  unified-pipeline
```

## Output

Results saved to:
- `knowledge_base/` - Knowledge extraction results
- `unified_output/` - Pipeline execution results
- `trained_models/` - Trained models (if training enabled)

## Troubleshooting

### Build Issues

```bash
# Rebuild without cache
docker-compose -f docker-compose.unified.yml build --no-cache
```

### Out of Memory

Increase memory limit in `docker-compose.unified.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increase from 8G
```

### Container Won't Start

Check logs:
```bash
docker-compose -f docker-compose.unified.yml logs unified-pipeline
```

## Status

✅ **Docker configuration complete**  
✅ **Dockerfile created**  
✅ **docker-compose.yml configured**  
⚠️ **Container build/testing needed**

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-18
