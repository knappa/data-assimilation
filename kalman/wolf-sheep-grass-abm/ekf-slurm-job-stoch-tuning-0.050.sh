#!/bin/bash
#SBATCH --job-name=st-0.050                   # Job name
#SBATCH --mail-type=ALL                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adam.knapp@ufl.edu      # Where to send mail
#SBATCH --nodes=1                           # Use one node (non-MPI)
#SBATCH --ntasks=1                          # Run a single task (non-MPI)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4gb                   # Memory per job
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --output=ekf-st-0.050-array_%A-%a.out  # Standard output and error log
#SBATCH --array=0-999                       # Array range
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users. 
pwd; hostname; date

#Set the number of runs that each SLURM task should do
PER_TASK=1

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END_NUM=$(( (SLURM_ARRAY_TASK_ID + 1) * PER_TASK ))

# Print the task and run range
echo "This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $(( END_NUM - 1 ))"

module load python3
cd /home/adam.knapp/blue_rlaubenbacher/adam.knapp/data-assimilation/kalman/wolf-sheep-grass-abm || exit
source venv/bin/activate

stoch_level=0.050

# Run the loop of runs for this task.
for (( run=START_NUM; run < END_NUM; run++ )); do
  echo "This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run"
  #Do your stuff here

  # grass
  PREFIX=g-$(printf %04d "$run")
  python3 ekf.py  \
    --prefix st-"$PREFIX"-$stoch_level \
    --measurements grass \
    --grass_r 0.01 \
    --matchmaker yes \
    --grid_width 51 \
    --grid_height 51 \
    --param_stoch_level $stoch_level

  # sheep
  PREFIX=s-$(printf %04d "$run")
  python3 ekf.py  \
    --prefix st-"$PREFIX"-$stoch_level \
    --measurements sheep \
    --sheep_r 0.01 \
    --matchmaker yes \
    --grid_width 51 \
    --grid_height 51 \
    --param_stoch_level $stoch_level

  # wolf
  PREFIX=w-$(printf %04d "$run")
  python3 ekf.py  \
    --prefix st-"$PREFIX"-$stoch_level \
    --measurements wolves \
    --wolf_r 0.01 \
    --matchmaker yes \
    --grid_width 51 \
    --grid_height 51 \
    --param_stoch_level $stoch_level

done

date


