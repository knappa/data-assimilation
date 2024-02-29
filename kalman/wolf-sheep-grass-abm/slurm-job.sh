#!/bin/bash
#SBATCH --job-name=wsg                 # Job name
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adam.knapp@ufl.edu # Where to send mail  
#SBATCH --nodes=1                      # Use one node (non-MPI)
#SBATCH --ntasks=1                     # Run a single task (non-MPI)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb              # Memory per job
#SBATCH --time=72:00:00                # Time limit hrs:min:sec
#SBATCH --output=wsg-array_%A-%a.out   # Standard output and error log
#SBATCH --array=0-6                    # Array range
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users. 
pwd; hostname; date

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID


module load python3
cd /home/adam.knapp/blue_rlaubenbacher/adam.knapp/data-assimilation/kalman/wolf-sheep-grass-abm

source venv/bin/activate

case $SLURM_ARRAY_TASK_ID in

    0)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-grass --measurements grass --matchmaker yes
	done
	;;

    1)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-sheep --measurements sheep --matchmaker yes
	done
	;;

    2)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-wolf --measurements wolves --matchmaker yes
	done
	;;

    3)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-wolfgrass --measurements wolves+grass --matchmaker yes
	done
	;;

    4)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-wolfsheep --measurements wolves+sheep --matchmaker yes
	done
	;;

    5)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-sheepgrass --measurements sheep+grass --matchmaker yes
	done
	;;
    
    6)
	for i in $(seq -f "%03g" 0 999)
	do
	    python3 ekf.py --prefix $i-wolfsheepgrass --measurements wolves+sheep+grass --matchmaker yes
	done
	;;

    *)
	echo "Whoops?"    
    esac

date
