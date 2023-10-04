#!/bin/bash

snakemake \
    --nolock \
    --cores 2 \
    --config \
    job_name=job_2023-10-04_test_v000 \
    n_iterations=2 \
    n_query=50 \
    nu=1.5 \
    scoring=mae \
    cutoff=20 \
    ylog=true \
    reason="small test without wasserstein, but with diversity and informativeness" \
    slurm_id=none
