#!/bin/bash

# Run all simulations for the study
METHODS=("landau" "fiebelkorn" "ar" "mann_lees")
N_EXP="n_experiments=1000"
USE_SBATCH=true  # whether to run the analyses using sbatch

# Load the necessary modules on the cluster
if [ $(hostname) = "bb-pg-login01.bear.cluster" ]; then
    echo Loading modules for Bluebear
    load_python-simulated_rhythmic_sampling.sh
fi

# Simulate different kinds of aperiodic behavior analyzed with each method
NOISE_PARAM_CHOICES=(
    "noise_method=fully_random"
    "noise_method=powerlaw, exponent=0"
    "noise_method=powerlaw, exponent=2"
    "noise_method=arma, ar_coefs=0.5, ma_coefs=0"
)
OSC_PARAMS="f_osc=0, osc_amp=0"
for METHOD in "${METHODS[@]}"
do
    for NOISE_PARAMS in "${NOISE_PARAM_CHOICES[@]}"
    do
        CALL_PARAMS="analysis_type=$METHOD, $N_EXP, $NOISE_PARAMS, $OSC_PARAMS"
        if [ "$USE_SBATCH" = true ]; then
            ./slurmit \
                "analysis.py \"$CALL_PARAMS\"" -t 2:00:00 -m 20G -d results/slurm
        else
            python3 analysis.py "$CALL_PARAMS"
        fi
    done
done

# How well does each method identify true behavioral oscillations?
NOISE_PARAMS="noise_method=powerlaw, exponent=2"
for METHOD in "${METHODS[@]}"
do
    for FREQ in $(seq 2 1 14)
    do
        for OSC_AMP in $(seq 0.1 0.1 0.6)
        do
            OSC_PARAMS="f_osc=$FREQ, osc_amp=$OSC_AMP"
            CALL_PARAMS="analysis_type=$METHOD, $N_EXP, $NOISE_PARAMS, $OSC_PARAMS"
            if [ "$USE_SBATCH" = true ]; then
                ./slurmit \
                    "analysis.py \"$CALL_PARAMS\"" -t 2:00:00 -m 20G -d results/slurm
            else
                python3 analysis.py "$CALL_PARAMS"
            fi
        done
    done
done
