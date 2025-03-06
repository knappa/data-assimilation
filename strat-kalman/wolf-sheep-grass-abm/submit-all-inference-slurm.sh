#!/bin/bash

sbatch slurm-grass-r-search-00.01.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-grass-r-search-00.10.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-grass-r-search-01.00.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-grass-r-search-10.00.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst

sbatch slurm-sheep-r-search-00.01.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-sheep-r-search-00.10.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-sheep-r-search-01.00.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-sheep-r-search-10.00.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst

sbatch slurm-wolf-r-search-00.01.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-wolf-r-search-00.10.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-wolf-r-search-01.00.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
sbatch slurm-wolf-r-search-10.00.sh | tr -s " " | cut -d " " -f 4 | xargs qos_to_burst
