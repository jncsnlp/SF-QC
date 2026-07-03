#!/bin/bash

python scripts/run.py \
  --task banking \
  --id_ratio 0.25 \
  --SF_ratio 0.1 \
  --llm qwen \
  --method SF-QC \
  --seed 15