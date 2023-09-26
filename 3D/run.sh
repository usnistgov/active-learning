#!/bin/bash

snakemake \
    --nolock \
    --cores 18 \
    --config \
    job_name=job_2023-09-25_wasserstein_v000 \
    n_iterations=18 \
    n_query=100 \
    nu=1.5 \
    scoring=mae \
    cutoff=20 \
    ylog=true \
    reason="debug wasserstein" \
    slurm_id=none
