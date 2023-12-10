#!/usr/bin/env python
#SBATCH --time=96:01:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=60             # Number of task (cores/ppn) per node
#SBATCH --job-name=R1_JA            # Name of batch job
#SBATCH --partition=biocrunch                 # Partition (queue) #pronto: biocrunch or speedy
#SBATCH --output=out_ktv.o%j            # Name of batch job output file
#SBATCH --error=err_ktv.e%j            # Name of batch job error file
#SBATCH --mail-user=yhqiang@iastate.edu  # Send email notifications
##SBATCH --mail-type=BEGIN           # Type of email notifications to send
#SBATCH --mail-type=END
#SBATCH --mem=200G
#SBATCH --array=1-8
###############################################################################

import os
jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
n_jobs = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
L = 16

cwd = os.getcwd()
os.system(f'/work/LAS/porth-lab/yhqiang/anaconda/bin/python {cwd}/JArray_R1_unphysical-v2.py {n_jobs} {jobid} {L}')
print(f'job id: {jobid}')