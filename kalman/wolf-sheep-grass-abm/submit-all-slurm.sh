#!/bin/bash

sbatch ekf-slurm-job-grass.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch ekf-slurm-job-sheep-grass.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch ekf-slurm-job-sheep.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch ekf-slurm-job-wolf-grass.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch ekf-slurm-job-wolf.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch ekf-slurm-job-wolf-sheep-grass.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch ekf-slurm-job-wolf-sheep.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
