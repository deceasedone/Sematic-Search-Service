#!/usr/bin/env bash
# Limit threads to prevent CPU oversubscription with multiple workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

uvicorn app.main:app --workers 4
