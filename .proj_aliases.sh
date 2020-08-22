#!/bin/bash

module load R/4.0.1 python/3.7.4 

# Bash aliases used in this project.
alias demeter_srun="srun --pty -p priority --mem 50G -c 5 -t 0-18:00 --x11 /bin/bash"
alias demeter_env="conda activate demeter2-stan && bash .proj_aliases.sh"
alias demeter_jl="jupyter lab --port=7010 --browser='none'"
alias demeter_sshlab='ssh -N -L 7010:127.0.0.1:7010'
