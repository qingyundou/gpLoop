#!/bin/bash

sbatch slurm_submit.peta4-icelake \
-J $job_name \
-A GALES-SL3-CPU \
-p icelake \
--nodes=1 \
--time=04:00:00 \
--mail-type=NONE \
--no-requeue