#!/bin/sh
#BSUB -J torch
#BSUB -o torch_%J.out
#BSUB -e torch_%J.err
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=30G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
# module load scipy/VERSION

# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!
# module purge
# module load python3/3.8.1

source my_env/bin/activate

python3 collectDataSetFromPickles.py


