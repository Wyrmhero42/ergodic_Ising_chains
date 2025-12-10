#!/bin/bash -l

#SBATCH --time=02:00:00
#SBATCH --job-name=Ising_tilted_L16
#SBATCH --export=NONE
#SBATCH --mail-user=leonhard.schmotzer@posteo.net --mail-type=ALL
#SBATCH --partition=spr1tb
#SBATCH --exclusive
#SBATCH --output=report/%x-%A_%a.out   # %A = array-jobID, %a = array index

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104

 #SBATCH --array=1-37

module load python/3.12-anaconda
#source ~/venvs/su2env/bin/activate


########################### definitions
divider () {
  printf '=%.0s' {1..50}
  printf '\n'
}
###########################

# set parameters
L=16

# divider
# echo "Job array index: ${SLURM_ARRAY_TASK_ID}"
# divider

# # define chunk size
# CHUNK=500

# # compute start and end for this job
# START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK ))
# END=$(( START + CHUNK ))

echo "Running: Ising_tilted_eigensystem.py $L"
divider

#python aperiodic/test_ensemble_evo.py $P $jmax $j1 $j2 $j3 $j4 $gsq $C1 $C2 $START $END
python Ising_tilted_eigensystem.py $L