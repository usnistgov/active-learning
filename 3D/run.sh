#!/bin/bash

snakemake \
    --nolock \
    --cores 20 \
    --config \
    job_name=job_2023-10-11_sinkhorn2_v000 \
    n_iterations=20 \
    n_query=50 \
    nu=1.5 \
    scoring=mae \
    cutoff=20 \
    ylog=true \
    reason="small test without wasserstein, but with diversity and informativeness" \
    slurm_id=none
