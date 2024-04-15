#!/usr/bin/env bash

./ekf.py \
  --measurements P_DAMPS T1IFN TNF IFNg IL1 IL6 IL8 IL10 IL12 IL18 extracellular_virus \
  --update-algorithm full-spatial \
  --uncertainty-P_DAMPS 0.01 \
  --uncertainty-T1IFN 0.01 \
  --uncertainty-TNF 0.01 \
  --uncertainty-IFNg 0.01 \
  --uncertainty-IL6 0.01 \
  --uncertainty-IL1 0.01 \
  --uncertainty-IL8 0.01 \
  --uncertainty-IL10 0.01 \
  --uncertainty-IL12 0.01 \
  --uncertainty-IL18 0.01 \
  --graphs \
  --grid_width 51 \
  --grid_height 51 \
  --prefix full \
  --predict to-next-kf-update \
  --time_span 2016 \
  --sample_interval 48 \
  # --verbose
