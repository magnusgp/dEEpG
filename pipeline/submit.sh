#!/bin/sh
#BSUB -J torch
#BSUB -o torch_%J.out
#BSUB -e torch_%J.err
#BSUB -q hpc
#BSUB -n 32
#BSUB -R "rusage[mem=20G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1440
#BSUB -u s204075@student.dtu.dk
#BSUB -B
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
# module load scipy/VERSION

# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!
source my_env/bin/activate

python3 pipelineMain.py --limit 200

